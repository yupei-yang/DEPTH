import argparse
import boto3
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import torch.distributed as dist

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.datasets.utils import zero_pad_sequences, exist_and_not_none


def is_rank_0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_logger(logger_path):
    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    if is_rank_0():
        file_handler = logging.FileHandler(logger_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def preprocess_data(
    data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None, tokenizer=None, apply_chat_template=False
) -> str:
    # custom dataset
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
    # nvidia/OpenMathInstruct-1
    elif exist_and_not_none(data, "question") and exist_and_not_none(data, "correct") and exist_and_not_none(data, "incorrect"):
        prompt = data["question"]
        chosen = data["correct"]
        reject = data["incorrect"]
    else:
        # Anthropic/hh-rlhf
        # tasksource/oasst1_pairwise_rlhf_reward
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ")
                    + "\nAssistant: "
                )
            chosen = data["chosen"]
            reject = data["rejected"]

            if apply_chat_template:
                
                chosen_full = tokenizer.apply_chat_template(chosen, tokenize=False)
                reject_full = tokenizer.apply_chat_template(reject, tokenize=False)
                chosen_end_inst_index = chosen_full.rfind("[/INST]") + len("[/INST]")
                reject_end_inst_index = reject_full.rfind("[/INST]") + len("[/INST]")
                prompt = chosen_full[:chosen_full.rfind("[/INST]") + len("[/INST]")]
                chosen = chosen_full[chosen_end_inst_index:]
                reject = reject_full[reject_end_inst_index:]

            input_template = None  # do not modified with input template again
        # lvwerra/stack-exchange-paired
        elif exist_and_not_none(data, "response_j"):
            prompt = data["question"]
            chosen = data["response_j"]
            reject = data["response_k"]
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"])
                return "\n".join(result)

            prompt = ""
            chosen = (
                data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
            )
            reject = (
                data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
            )
            chosen = process_chatbot_arena_conversations(chosen)
            reject = process_chatbot_arena_conversations(reject)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
        # damo/CValues-Comparison https://www.modelscope.cn/datasets/damo/CValues-Comparison/quickstart
        elif exist_and_not_none(data, "pos_resp") and exist_and_not_none(data, "neg_resp"):
            prompt = data["prompt"]
            chosen = data["pos_resp"]
            reject = data["neg_resp"]
        else:
            raise ValueError("Unknown reward dataset")

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, chosen, reject, margin


class EvalDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, strategy, input_template=None, is_dpo=False, apply_chat_template=False):
        super().__init__()
        self.is_dpo = is_dpo
        self.prompts, self.chosens, self.rejects = [], [], []
        self.prompt_ids_lens = [] if is_dpo else []
        self.margins = [] if not is_dpo else []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        prompt_key = getattr(strategy.args, "prompt_key", None)
        chosen_key = getattr(strategy.args, "chosen_key", None)
        rejected_key = getattr(strategy.args, "rejected_key", None)

        for data in tqdm(dataset, disable=not strategy.is_rank_0()):
            prompt, chosen, reject, margin = preprocess_data(data, input_template, prompt_key, chosen_key, rejected_key, tokenizer, apply_chat_template)
            if is_dpo:
                prompt_token = tokenizer(prompt, max_length=max_length, padding=False, truncation=True, return_tensors="pt")
                if prompt_token["attention_mask"].sum().item() >= max_length - 2:
                    continue
                self.prompt_ids_lens.append(prompt_token["attention_mask"].sum().item())
            else:
                self.margins.append(margin)
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)

    def __len__(self):
        return len(self.chosens)

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]
        extra = self.prompt_ids_lens[idx] if self.is_dpo else self.margins[idx]

        chosen_text = prompt + chosen + " " + self.tokenizer.eos_token
        reject_text = prompt + reject + " " + self.tokenizer.eos_token

        chosen_token = self.tokenizer(chosen_text, max_length=self.max_length, padding=False, truncation=True, return_tensors="pt")
        reject_token = self.tokenizer(reject_text, max_length=self.max_length, padding=False, truncation=True, return_tensors="pt")

        # ensure EOS not truncated
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
            chosen_text,
            reject_text,
        )

    def collate_fn(self, item_list):
        chosen_ids, chosen_masks, reject_ids, reject_masks = [], [], [], []
        extras, chosen_texts, rejected_texts = [], [], []

        for c_id, c_mask, r_id, r_mask, extra, c_text, r_text in item_list:
            chosen_ids.append(c_id)
            chosen_masks.append(c_mask)
            reject_ids.append(r_id)
            reject_masks.append(r_mask)
            extras.append(extra)
            chosen_texts.append(c_text)
            rejected_texts.append(r_text)

        return (
            zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id),
            zero_pad_sequences(chosen_masks),
            zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id),
            zero_pad_sequences(reject_masks),
            extras,
            chosen_texts,
            rejected_texts,
        )

def concatenated_forward(model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
    input_ids, att_masks = concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask)
    all_values, _, _, _ = model(
        input_ids, attention_mask=att_masks, return_output=True
    )
    chosen_rewards = all_values[: chosen_ids.shape[0]]
    rejected_rewards = all_values[chosen_ids.shape[0]:]
    return chosen_rewards, rejected_rewards


def concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
    def pad_to_length(tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            # left pad
            return torch.cat(
                [
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    tensor,
                ],
                dim=dim,
            )

    max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
    inputs_ids = torch.cat(
        (
            pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
        ),
        dim=0,
    )
    max_length = max(c_mask.shape[1], r_mask.shape[1])
    att_masks = torch.cat(
        (pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0
    )
    return inputs_ids, att_masks

def evaluate(strategy, model, tokenizer, dataloader, logger):
    step_bar = tqdm(range(len(dataloader)), disable=not strategy.is_rank_0())
    acc, rewards = 0, []

    with torch.no_grad():
        for idx, (c_ids, c_mask, r_ids, r_mask, margin, c_texts, r_texts) in enumerate(dataloader):
            c_ids, c_mask = c_ids.squeeze(1).to(torch.cuda.current_device()), c_mask.squeeze(1).to(torch.cuda.current_device())
            r_ids, r_mask = r_ids.squeeze(1).to(torch.cuda.current_device()), r_mask.squeeze(1).to(torch.cuda.current_device())

            c_rewards, r_rewards = concatenated_forward(model, tokenizer, c_ids, c_mask, r_ids, r_mask)
            acc_batch = (c_rewards > r_rewards).float().mean().item()
            acc += acc_batch
            rewards.extend([c_rewards.flatten(), r_rewards.flatten()])

            if strategy.is_rank_0():
                logger.info(f"Step [{idx + 1}/{len(dataloader)}] | Batch Acc: {acc_batch:.4f} | "
                            f"Chosen Mean: {c_rewards.mean().item():.4f} | Rejected Mean: {r_rewards.mean().item():.4f}")

            step_bar.update()

    acc_mean = acc / len(dataloader)
    rewards = torch.cat(rewards).float()
    rewards = strategy.all_gather(rewards)
    if strategy.is_rank_0():
        logger.info(f"Final Accuracy: {acc_mean:.4f}")
        logger.info(f"Final Reward Mean: {rewards.mean().item():.4f}")
        logger.info(f"Final Reward Std: {rewards.std().clamp(min=1e-8).item():.4f}")


def eval_rm(args):
    logger = setup_logger(args.save_dir)
    strategy = get_strategy(args)
    strategy.setup_distributed()

    if strategy.is_rank_0():
        boto3.client("s3")

    model = get_llm_for_sequence_regression(
        args.reward_pretrain, "reward",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=False),
    ).eval().to(torch.cuda.current_device())

    tokenizer = get_tokenizer(args.reward_pretrain, model, "left", strategy)
    strategy.print(model)

    if strategy.is_rank_0():
        logger.info(f"Reward normalization: {args.normalize_reward}")
        logger.info(f"Reward mean: {getattr(model, 'mean', 'N/A')} | std: {getattr(model, 'std', 'N/A')}")

    raw_data, _ = blending_datasets(args.eval_data_path, args.eval_data_probs, strategy, args.seed, max_count=5000000, stopping_strategy="all_exhausted")
    raw_data = raw_data.select(range(min(args.max_samples, len(raw_data))))

    dataset = EvalDataset(raw_data, tokenizer, args.max_len, strategy, input_template=args.input_template, apply_chat_template=args.apply_chat_template)

    dataloader = strategy.setup_dataloader(dataset, args.batch_size, pin_memory=True, shuffle=True, collate_fn=dataset.collate_fn)

    strategy.print(f"Eval dataset size: {len(dataset)}, Dataloader steps: {len(dataloader)}")
    evaluate(strategy, model, tokenizer, dataloader, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_path", type=str, default="dataset/SciERC/train_data/arrow_files/RM")
    parser.add_argument("--eval_data_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--reward_pretrain", type=str, default="ckpt/SciERC/RM/MERGED/V0")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--normalize_reward", action="store_true", default=True)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="src/eval/eval_causal_rm/logs/eval-on-train.log")

    args = parser.parse_args()
    eval_rm(args)

# deepspeed src/eval/eval_causal_rm/eval.py