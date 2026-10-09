#!/bin/bash

# ==========================================
# Environment Variable Configuration
# ==========================================

# 1. Specify the GPU IDs to use
export CUDA_VISIBLE_DEVICES="3,4"

# 2. VRAM fragmentation optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 3. Other settings
export WANDB_MODE="disabled"
export PYTHONWARNINGS="ignore"

# ==========================================
# Inference Parameter Settings
# ==========================================

# 1. Result saving path
ROOT_PATH="test_result"

# ===== Configure specific prediction steps to save physical quantities separately =====
# For example, to save the results for steps 1, 3, and 5
SAVE_STEPS="5 10"

# 2. Inference specific parameters
# Define multiple models using a Bash array, separated by spaces or newlines
MODELS=(
        "diagno_e128"
)

INFER_STEPS=10
BATCH_SIZE=16
NUM_EVAL_SAMPLES=256

# ==========================================
# Execution Commands
# ==========================================

echo "---------------------------------------"
echo "Starting Spherical SWE batch autoregressive inference task"
echo "Experiment root path: $ROOT_PATH"
echo "Inference steps: $INFER_STEPS"
echo "Specific steps to save: $SAVE_STEPS"
echo "Using device: Physical GPU $CUDA_VISIBLE_DEVICES"
echo "VRAM optimization: $PYTORCH_CUDA_ALLOC_CONF"
echo "Batch Size: $BATCH_SIZE"
echo "---------------------------------------"

mkdir -p "$ROOT_PATH"

# Iterate through the model list and execute inference
for MODEL_NAME in "${MODELS[@]}"; do
    echo "======================================="
    echo "Inferring model: $MODEL_NAME"
    echo "======================================="
    
    # Dynamically construct the finetuned weight path for the current model to prevent loading incorrect weights
    FINETUNED_MODEL_PATH="${ROOT_PATH}/${MODEL_NAME}/best_model_finetune.pt"
    
    FINETUNED_ARG=""
    if [ -f "$FINETUNED_MODEL_PATH" ]; then
        FINETUNED_ARG="--pretrained_path $FINETUNED_MODEL_PATH"
        echo "Finetuned weights path detected, loading: $FINETUNED_MODEL_PATH"
    else
        echo "Warning: best_model_finetune.pt not found for this model, handing over to Python script"
    fi

    torchrun --nproc_per_node=2 --master_port=29562 infer.py \
        --root_path "$ROOT_PATH" \
        --model_name "$MODEL_NAME" \
        --infer_steps $INFER_STEPS \
        --batch_size $BATCH_SIZE \
        --num_eval_samples $NUM_EVAL_SAMPLES \
        $FINETUNED_ARG \
        --save_steps $SAVE_STEPS

done

echo "---------------------------------------"
echo "All model inference scripts executed successfully."
