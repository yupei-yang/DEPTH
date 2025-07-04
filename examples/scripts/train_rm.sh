set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
     --save_path checkpoint/SciERC/RM/V0 \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 64 \
     --micro_train_batch_size 1 \
     --pretrain meta-llama/Llama-3.1-8B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset dataset/SciERC/train_data/arrow_files/RM \
     --prompt_key question \
     --chosen_key correct \
     --rejected_key incorrect \
     --flash_attn \
     --packing_samples \
     --gradient_checkpointing \
     --use_wandb {wandb_token}
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi