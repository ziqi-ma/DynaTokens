SHELL := /bin/bash
.RECIPEPREFIX := >
.DEFAULT_GOAL := help

ENV_NAME ?= dynatoken
HF_TOKEN ?=
DATA_BASE ?=
CKPT ?=

.PHONY: help setup setup-flash-attn models submodules test prepare train infer vbench worldscore camera

help: ## Show available developer commands
>@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create or update the main Conda environment
>ENV_NAME=$(ENV_NAME) bash scripts/setup_dev.sh

setup-flash-attn: ## Create/update the environment and install FlashAttention
>ENV_NAME=$(ENV_NAME) bash scripts/setup_dev.sh --flash-attn

models: ## Download HY-WorldPlay and FLUX checkpoints (requires HF_TOKEN)
>@test -n "$(HF_TOKEN)" || (echo "Set HF_TOKEN=<token>" >&2; exit 2)
>ENV_NAME=$(ENV_NAME) HF_TOKEN=$(HF_TOKEN) bash scripts/setup_dev.sh --download-models

submodules: ## Initialize and update git submodules
>git submodule update --init --recursive

test: ## Run the CPU-safe pytest suite
>python -m pytest $(PYTEST_ARGS)

prepare: ## Encode training data; pass PREPARE_ARGS="..."
>PYTHONPATH=model/hywp python dynatokens/prepare_training_data.py $(PREPARE_ARGS)

train: ## Train tokens; pass DATA_BASE=... and optional TRAIN_ENV="..."
>DATA_BASE=$(DATA_BASE) $(TRAIN_ENV) bash dynatokens/train.sh

infer: ## Render trajectories; pass CKPT=... and optional INFER_ENV="..."
>CKPT=$(CKPT) $(INFER_ENV) bash dynatokens/inference.sh

vbench: ## Set up the VBench evaluation environment
>bash metrics/vbench/setup_env.sh

worldscore: ## Print the WorldScore evaluation command
>@echo "conda activate worldscore"
>@echo "python metrics/motion/eval_motion.py <rendered-eval-directory>"

camera: ## Set up the ViPE camera evaluation environment
>bash metrics/camera/setup.sh
