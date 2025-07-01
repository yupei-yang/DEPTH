# DEPTH

## Installation (Python 3.10, CUDA 12.1, Torch 2.2)

```bash
conda create -n llm4ie python=3.10
conda activate llm4ie

# Install build tools
pip install packaging ninja

# Install PyTorch manually (CUDA 12.1 support)
pip install --no-cache-dir torch==2.2.0 torchvision torchaudio

# (Optional) If you're using environment modules:
# module load cuda/12.1
```

### 2. Install FlashAttention

This project uses a CUDA-compiled `.whl` of FlashAttention.
Install it manually from your local path (relative to project root):

```bash
pip install flash_attn/flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```

> Make sure the `.whl` file is downloaded from:
> [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)


### 3. Install Python dependencies and this project

Install the current repo (in editable mode):

```bash
cd src/train/build_scripts/
./build_openrlhf.sh
```

You're now ready to use `DEPTH`!
