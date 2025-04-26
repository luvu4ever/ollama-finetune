#!/bin/bash

# Create necessary directories
mkdir -p outputs
mkdir -p data
mkdir -p models

# Maximum number of attempts
MAX_ATTEMPTS=10
ATTEMPT=1

# Function to find the latest checkpoint
find_latest_checkpoint() {
    CHECKPOINTS_DIR="./outputs"
    
    # Look for checkpoint directories
    LATEST_CHECKPOINT=$(find "$CHECKPOINTS_DIR" -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n 1)
    
    # If no checkpoint-* directories found, try recovery checkpoints
    if [ -z "$LATEST_CHECKPOINT" ]; then
        LATEST_CHECKPOINT=$(find "$CHECKPOINTS_DIR" -type d -name "recovery_step_*" 2>/dev/null | sort -V | tail -n 1)
    fi
    
    # If still no checkpoints, try emergency checkpoint
    if [ -z "$LATEST_CHECKPOINT" ]; then
        if [ -d "$CHECKPOINTS_DIR/emergency_checkpoint" ]; then
            LATEST_CHECKPOINT="$CHECKPOINTS_DIR/emergency_checkpoint"
        fi
    fi
    
    echo "$LATEST_CHECKPOINT"
}

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "Torch not available to clear cache"

# Initial run with no checkpoint
echo "===== TRAINING ATTEMPT $ATTEMPT ====="
/app/full_training_with_resumption.sh
EXIT_CODE=$?

# Loop while training fails and we haven't exceeded maximum attempts
while [ $EXIT_CODE -ne 0 ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(find_latest_checkpoint)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Found checkpoint: $LATEST_CHECKPOINT"
        echo "===== RESUMING TRAINING ATTEMPT $ATTEMPT ====="
        
        # Clear GPU cache before restarting
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "Torch not available to clear cache"
        
        # Wait to give system time to recover
        echo "Waiting 30 seconds before resuming..."
        sleep 30
        
        # Resume training from the latest checkpoint
        ./full_training_with_resumption.sh --resume "$LATEST_CHECKPOINT"
        EXIT_CODE=$?
    else
        echo "No checkpoint found. Cannot resume."
        exit 1
    fi
done

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== TRAINING COMPLETED SUCCESSFULLY AFTER $ATTEMPT ATTEMPTS ====="
    exit 0
else
    echo "===== TRAINING FAILED AFTER $ATTEMPT ATTEMPTS ====="
    echo "You may want to adjust your training parameters."
    exit 1
fi