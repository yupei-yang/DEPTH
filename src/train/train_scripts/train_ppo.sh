set -x

# RLHF Prompts: openmath instruct prompts
read -r -d '' training_commands <<EOF
./examples/train_ppo.py \
    --pretrain meta-llama/Llama-3.2-3B-Instruct \
    --reward_pretrain ckpt/SciERC/RM/MERGED/V0 \
    --save_path ckpt/SciERC/Actor/LORA/V0 \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 8 \
    --max_epochs 1 \
    --prompt_max_len 4096 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.02 \
    --prompt_data examples/data/SciERC/Actor \
    --prompt_data_probs 1.0 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules "value_head,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --use_wandb 21fa40bfdc7846ab6ff6a354af592107d0ea761d \
    --wandb_project LLM4IE-RLHF \
    --wandb_org cmach-yyp \
    --max_samples 100000
EOF

if [[ ${1} != "slurm" ]]; then
    CUDA_VISIBLE_DEVICES=0,1 deepspeed $training_commands
fi