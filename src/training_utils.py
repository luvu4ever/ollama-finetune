import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import unsloth_train
import os

def get_training_args(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    save_total_limit=3
):
    """
    Create training arguments for the trainer.
    
    Args:
        output_dir (str): Directory to save outputs
        per_device_train_batch_size (int): Batch size per device
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        learning_rate (float): Peak learning rate
        num_train_epochs (int): Number of training epochs
        save_steps (int): Save checkpoint every X steps
        eval_steps (int): Run evaluation every X steps
        save_total_limit (int): Maximum number of checkpoints to keep
        
    Returns:
        TrainingArguments: Configuration for training
    """
    return TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=False,  # Force fp16 for memory savings
        bf16=True,  # Disable bf16 to ensure fp16 is used
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        
        logging_steps=20,
        logging_strategy="steps",
        
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        
        load_best_model_at_end=True,
        save_only_model=False,  # Changed to False to store optimizer state for resuming
        save_total_limit=save_total_limit,
        
        # Memory optimization settings balanced with checkpointing needs
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        # Allow checkpoint resumption
        resume_from_checkpoint=True,
    )

def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    compute_metrics,
    training_args,
    max_seq_length=1024  # Using full sequence length as requested
):
    """
    Create and configure the SFT trainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        compute_metrics: Function to compute evaluation metrics
        training_args: Training arguments
        max_seq_length (int): Maximum sequence length
        
    Returns:
        SFTTrainer: Configured trainer
    """
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        compute_metrics=compute_metrics,
        args=training_args,
    )
    
    return trainer

def train_model(trainer, resume_from=None):
    """
    Train the model using unsloth optimizations with checkpoint resumption.
    
    Args:
        trainer: Configured trainer
        resume_from: Path to checkpoint to resume from
        
    Returns:
        dict: Training statistics
    """
    print("Starting training...")
    
    # Check if resuming from checkpoint
    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        # Explicitly set the resumption path
        trainer.args.resume_from_checkpoint = resume_from
    else:
        print("Starting training from scratch")
    
    try:
        # Train with unsloth optimizations
        trainer_stats = unsloth_train(trainer)
        print("Training completed successfully!")
    except RuntimeError as e:
        # Check if this is an OOM error
        if "CUDA out of memory" in str(e):
            print("\n\n==== CUDA OUT OF MEMORY ERROR DETECTED ====")
            print("This is normal when using max_seq_length=1024.")
            print("You should resume training from the last checkpoint.")
            
            # Find the most recent checkpoint
            checkpoints = [d for d in os.listdir(trainer.args.output_dir) 
                          if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = os.path.join(trainer.args.output_dir, checkpoints[-1])
                print(f"\nLatest checkpoint: {latest_checkpoint}")
                print(f"\nTo resume training, run with:")
                print(f"  --resume_from_checkpoint {latest_checkpoint}")
            else:
                print("\nNo checkpoints found. You might need to adjust memory settings.")
                
            # Re-raise the exception
            raise
        else:
            # Re-raise other exceptions
            raise
            
    return trainer_stats