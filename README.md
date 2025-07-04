# DEPTH

DEPTH is a project based on [OpenRLHF v0.8.5](https://github.com/OpenRLHF/OpenRLHF), designed for RLHF (Reinforcement Learning from Human Feedback) and LLM training.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yupei-yang/DEPTH.git
```

### 2. Enter the Project Directory

```bash
cd DEPTH
```

### 3. Launch the Docker Container

Make sure you have NVIDIA drivers, Docker, and nvidia-docker installed.

```bash
docker run --runtime=nvidia -it --shm-size="10g" --cap-add=SYS_ADMIN \
  --name depth \
  -v $PWD:/workspace/DEPTH \
  nvcr.io/nvidia/pytorch:25.02-py3 bash
```

### 4. Install Dependencies

Inside the docker container, run:

1. Uninstall conflicting packages:

    ```bash
    pip uninstall xgboost transformer_engine flash_attn pynvml -y
    ```

2. (Optional) Switch to Tsinghua PyPI mirror for faster download in China:

    ```bash
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    ```

3. Install core requirements:

    ```bash
    pip install openrlhf[vllm]
    ```

4. Install this project in editable mode:

    ```bash
    pip install -e .
    ```

---

## Training & Usage

### 1. Prepare Environment Variables

Before training, set your HuggingFace and Weights & Biases credentials (obtain your own tokens and keys):

```bash
export HF_TOKEN=<your_huggingface_token>
export HF_ENDPOINT=<your_huggingface_endpoint>
export WANDB_BASE_URL=<your_wandb_base_url>
export WANDB_API_KEY=<your_wandb_api_key>
```

---

### 2. Train Reward Model (RM)

Inside the Docker container:

```bash
bash examples/scripts/train_rm.sh
```

---

### 3. Train with PPO

Start the Ray cluster and then run the PPO training script:

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
bash examples/scripts/train_ppo.sh
```

---

## Notes

- This project is based on [OpenRLHF v0.8.5](https://github.com/OpenRLHF/OpenRLHF).
- Please ensure NVIDIA drivers and nvidia-docker are properly configured before running.
- Adjust paths and GPU numbers according to your own server and setup.

---

## Reference

- [OpenRLHF v0.8.5](https://github.com/OpenRLHF/OpenRLHF)
- [NVIDIA PyTorch Docker Images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
