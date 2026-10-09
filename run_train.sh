#!/bin/bash

# ==========================================
# Environment Variable Configuration
# ==========================================
export PYTHONWARNINGS="ignore"

# ==========================================
# Training Parameters Configuration
# ==========================================
ROOT_PATH="test_result/"

MODELS="diagno_e128 diagno_e64"

PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS=100

BATCH_SIZE=4

PRETRAIN_LR=1e-3
FINETUNE_LR=1e-5

AMP_MODE="none"

RESUME_FLAG="--resume"

# ==========================================
# Execution Command
# ==========================================
echo "---------------------------------------"
echo "Starting Spherical SWE Training Task"
echo "Output Path: $ROOT_PATH"
echo "Models: $MODELS"
echo "Target Device: Physical GPU $CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE"
echo "Pretrain LR: $PRETRAIN_LR"
echo "Finetune LR: $FINETUNE_LR"
echo "Precision: $AMP_MODE"
echo "---------------------------------------"

mkdir -p "$ROOT_PATH"

export CUDA_VISIBLE_DEVICES="0"
torchrun --nproc_per_node=1 --master_port=29512 train.py \
    --root_path "$ROOT_PATH" \
    --models $MODELS \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --finetune_epochs $FINETUNE_EPOCHS \
    --batch_size $BATCH_SIZE \
    --pretrain_lr $PRETRAIN_LR \
    --finetune_lr $FINETUNE_LR \
    --amp_mode "$AMP_MODE" \
    $RESUME_FLAG

echo "---------------------------------------"
echo "Training script execution completed."