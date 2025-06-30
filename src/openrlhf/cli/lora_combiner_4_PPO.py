import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_actor_lora(base_model_name, lora_path, output_dir):
    # ========== 加载基础模型 ==========
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        quantization_config=None,
        torch_dtype="auto",
        cache_dir=""
    )
    
    # ========== 加载Tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # ========== 加载并合并LoRA ==========
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype="auto"
    )
    merged_model = lora_model.merge_and_unload()
       
    # ========== 保存完整模型 ==========
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        state_dict=merged_model.state_dict()
    )
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 合并完成，模型已保存到 {output_dir}")


if __name__ == "__main__":
    merge_actor_lora(
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        lora_path="ckpt/SciERC/Actor/LORA/V0",
        output_dir="ckpt/SciERC/Actor/MERGED/V0"
    )
