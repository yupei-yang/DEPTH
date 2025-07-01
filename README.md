# DEPTH

To install:
```
conda create -n llm4ie python=3.10

conda activate llm4ie

pip install packaging ninja

pip install --no-cache-dir torch==2.2.0 torchvision torchaudio

(module load cuda/12.1)

(pip install --force-reinstall --no-cache-dir numpy==1.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple)
```

Download flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl from https://github.com/Dao-AILab/flash-attention/releases?page=4

```
pip install flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```

Revise requirements.txt as:
```
bitsandbytes
datasets
deepspeed==0.13.2
einops
flash-attn==2.4.3.post1
isort
jsonlines
loralib
optimum
peft
ray[default]
torch
torchmetrics
tqdm
transformers==4.51.3
transformers-stream-generator==0.0.5
wandb
boto3
awscli
```

Then, run
```
cd src/train/build_scripts/build_openrlhf.sh
./build_openrlhf.sh
```