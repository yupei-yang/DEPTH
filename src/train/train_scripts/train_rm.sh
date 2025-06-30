set -x 

# Do training
read -r -d '' training_commands <<EOF
./examples/train_rm.py \
     --save_path ckpt/SciERC/RM/LORA/V0 \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 64 \
     --micro_train_batch_size 16 \
     --pretrain meta-llama/Llama-3.2-3B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 2 \
     --learning_rate 9e-6 \
     --dataset examples/data/SciERC/RM \
     --dataset_probs 1.0 \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --flash_attn \
     --gradient_checkpointing \
     --lora_rank 8 \
     --lora_alpha 16 \
     --use_wandb 21fa40bfdc7846ab6ff6a354af592107d0ea761d \
     --wandb_project LLM4IE \
     --wandb_org cmach-yyp \
     --reward_model_strategy vanilla \
     --contrastive_strategy cosine \
     --value_head_strategy linear
EOF


if [[ ${1} != "slurm" ]]; then
     CUDA_VISIBLE_DEVICES=0,1 deepspeed $training_commands
fi
