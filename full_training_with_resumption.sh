#!/bin/bash

# Create output directories if they don't exist
mkdir -p outputs
mkdir -p data
mkdir -p models

# Check if a checkpoint path is provided
if [ "$1" == "--resume" ] && [ -n "$2" ]; then
    RESUME_PATH="$2"
    echo "Resuming training from checkpoint: $RESUME_PATH"
    RESUME_FLAG="--resume_from_checkpoint $RESUME_PATH"
else
    RESUME_FLAG=""
    echo "Starting new training session"
fi

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "Torch not available to clear cache"

# Run with full sequence length and all data
echo "Starting training with full sequence length (1024)..."
python /app/run_training_with_checkpoints.py \
  --max_seq_length 1024 \
  --batch_size 1 \
  --grad_accum 4 \
  --lora_r 16 \
  --epochs 3 \
  --save_steps 100 \
  --eval_steps 100 \
  --checkpoint_steps 50 \
  --save_total_limit 3 \
  --load_in_4bit \
  --save_on_interrupt \
  $RESUME_FLAG

# Capture exit code
EXIT_CODE=$?

# Inform user of training status
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training exited with code: $EXIT_CODE"
    if [ -d "outputs/emergency_checkpoint" ]; then
        echo "An emergency checkpoint was saved. You can resume with:"
        echo "./full_training_with_resumption.sh --resume outputs/emergency_checkpoint"
    fi
fi

# Return the same exit code
exit $EXIT_CODE