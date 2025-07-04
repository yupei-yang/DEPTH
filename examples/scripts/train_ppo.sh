set -x 

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
      "working_dir": "/workspace/DEPTH",
      "excludes": [
        "/dataset/**",
        "/checkpoint/**",
        "/wandb/**",
        "/.git/**"
      ]
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --reward_pretrain /workspace/DEPTH/checkpoint/SciERC/RM/V0 \
   --save_path /workspace/DEPTH/checkpoint/SciERC/Actor/V0 \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --reward_prompt_max_len 2048 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /workspace/DEPTH/dataset/SciERC/train_data/arrow_files/Actor \
   --input_key question \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb {wandb_token}

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward