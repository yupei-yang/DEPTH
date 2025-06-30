import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from openrlhf.models import get_llm_for_sequence_regression


def apply_lora(model_name_or_path, lora_path, output_path, bf16=True):
    # ================== 加载基础模型 ==================
    base = get_llm_for_sequence_regression(
        model_name_or_path,
        "reward",
        init_value_head=True,
        bf16=bf16,
        load_in_4bit=False,
        lora_rank=0,
        value_head_strategy="linear",
        cache_dir=""
    )
    
    # ================== 加载Tokenizer ==================
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # ================== 加载并合并LoRA ==================
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32
    )
    merged_model = lora_model.merge_and_unload()
       
    # ================== 保存模型 ==================
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        state_dict=merged_model.state_dict()  # 显式保存所有参数
    )
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    apply_lora(
        model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        lora_path="ckpt/SciERC/RM/LORA/V0",
        output_path="ckpt/SciERC/RM/MERGED/V0",
    )
