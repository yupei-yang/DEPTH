import argparse
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist

from openrlhf.datasets import RewardDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_strategy, get_tokenizer


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


def concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask, tokenizer):
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """

    def pad_to_length(tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            # left pad
            return torch.cat(
                [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
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
    att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
    return inputs_ids, att_masks


def concatenated_forward(model, chosen_ids, c_mask, reject_ids, r_mask, tokenizer):
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    input_ids, att_masks = concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask, tokenizer)
    all_values, _ = model(input_ids, attention_mask=att_masks, return_output=True)
    chosen_rewards = all_values[: chosen_ids.shape[0]]
    rejected_rewards = all_values[chosen_ids.shape[0] :]
    return chosen_rewards, rejected_rewards


def evaluate(strategy, model, tokenizer, dataloader, logger):
    step_bar = tqdm(range(len(dataloader)), disable=not strategy.is_rank_0())
    acc, rewards = 0, []

    with torch.no_grad():
        for idx, (c_ids, c_mask, r_ids, r_mask, margin) in enumerate(dataloader):
            c_ids, c_mask = c_ids.squeeze(1).to(torch.cuda.current_device()), c_mask.squeeze(1).to(torch.cuda.current_device())
            r_ids, r_mask = r_ids.squeeze(1).to(torch.cuda.current_device()), r_mask.squeeze(1).to(torch.cuda.current_device())

            c_rewards, r_rewards = concatenated_forward(model, c_ids, c_mask, r_ids, r_mask, tokenizer)
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

    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.reward_pretrain,
        "reward",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
    )

    strategy.print(model)
    strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
    strategy.print("mean: {}, std {}".format(model.mean, model.std))

    if strategy.args.ref_reward_offload:
        model._offload = True

    model = strategy.prepare(model, is_rlhf=True)
    model.eval()

    # configure tokenizer
    tokenizer = get_tokenizer(args.reward_pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare for data and dataset
    eval_data = blending_datasets(
        args.eval_data_path,
        None,
        strategy,
        dataset_split=args.dataset_split,
    )

    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    eval_dataset = RewardDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
    )

    # prepare dataloader
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.batch_size,
        True,
        False,
        eval_dataset.collate_fn,
    )

    evaluate(strategy, model, tokenizer, eval_dataloader, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_path", type=str, default="dataset/SciERC/train_data/arrow_files/RM", help="HF dataset name or path")
    parser.add_argument(
        "--eval_data_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--reward_pretrain", type=str, default="checkpoint/SciERC/RM/LORA/V0", help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--packing_samples", action="store_true", default=True)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=True, help="Enable FlashAttention2")
    parser.add_argument("--bf16", action="store_true", default=True, help="Enable bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--max_len", type=int, default=2048, help="deprecated max_len")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=1024, help="Max number of samples")
    parser.add_argument("--prompt_key", type=str, default="question")
    parser.add_argument("--chosen_key", type=str, default="correct")
    parser.add_argument("--rejected_key", type=str, default="incorrect")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--zero_stage", type=int, default=3, help="DeepSpeed ZeRO stage")
    parser.add_argument("--save_dir", type=str, default="examples/eval/eval_causal_rm/logs/eval-on-train.log")

    args = parser.parse_args()
    eval_rm(args)

# deepspeed examples/eval/eval_causal_rm/eval_rm.py