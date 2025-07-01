FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/environment.yml
COPY . /workspace
WORKDIR /workspace

RUN conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "llm4ie", "/bin/bash", "-c"]

RUN chmod +x src/train/train_scripts/*.sh

# 用 pip 安装本地 wheel（路径要写对，/workspace是你的工程根目录）
RUN conda run -n llm4ie pip install ./flash_attn/flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

CMD ["bash"]