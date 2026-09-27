#!/bin/bash
# Environment for metrics/vbench/eval_vbench2.py (VBench-2.0 DSR / Object Presence / MOU).
# Usage: bash metrics/vbench/setup_env.sh   (needs `git submodule update --init third_party/VBench`)
#        ENV_NAME=<name> bash metrics/vbench/setup_env.sh   to use another env name

set -e

ENV_NAME="${ENV_NAME:-vbench2}"

echo "Creating conda environment: $ENV_NAME"
conda create -y -n "$ENV_NAME" python=3.10 pip

echo "Activating environment and installing packages..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# PyTorch with CUDA 12.1 (compatible with driver >= 525)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Core ML/vision deps
pip install \
    transformers==4.51.0 \
    accelerate==1.2.1 \
    tokenizers==0.21.1 \
    safetensors==0.5.2 \
    sentencepiece==0.2.0 \
    peft==0.14.0 \
    huggingface-hub==0.30.0 \
    bitsandbytes==0.42.0

# Video + image processing
pip install \
    decord==0.6.0 \
    Pillow==10.2.0 \
    opencv-python-headless==4.10.0.84 \
    imageio==2.37.0 \
    scenedetect==0.6.7.1 \
    gdown==6.0.0

# LLaVA-NeXT utility deps
pip install \
    einops==0.8.0 \
    einops-exts==0.0.4 \
    qwen-vl-utils==0.0.11 \
    numpy==1.26.3 \
    tqdm==4.67.1 \
    packaging \
    requests \
    regex \
    filelock \
    PyYAML \
    timm==0.4.12 \
    google-generativeai==0.8.6

# LLaVA-NeXT (pinned to the commit used in the full vbench2 env)
pip install "git+https://github.com/LLaVA-VL/LLaVA-NeXT@79ef45a6d8b89b92d7a8525f077c3a3a9894a87d#egg=llava"

# flash-attn, only needed for eval_vbench2.py --attn flash_attention_2 (optional, skip if build fails)
pip install flash-attn==2.7.2.post1 --no-build-isolation || \
    echo "WARNING: flash-attn installation failed — only --attn flash_attention_2 needs it."

# Judge models, in the location VBench-2.0 expects (~/.cache/vbench2)
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --local-dir ~/.cache/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/.cache/vbench2/Qwen/Qwen2.5-7B-Instruct

echo "Environment '$ENV_NAME' ready: conda activate $ENV_NAME"
