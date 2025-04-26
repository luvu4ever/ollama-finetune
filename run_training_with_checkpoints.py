import os
import argparse
import signal
import sys
import torch
import gc
from src.data_utils import load_data, split_data, prepare_datasets
from src.model_utils import load_model, setup_peft_model, save_model, compute_metrics
from src.training_utils import get_training_args, setup_trainer, train_model

# Global variables to handle graceful shutdown
trainer = None
model = None
tokenizer = None
save_on_interrupt = True

def signal_handler(sig, frame):
    """Handle interrupt signals by saving the model before exiting"""
    if save_on_interrupt and trainer is not None and model is not None and tokenizer is not None:
        print("\n\nInterrupt received. Saving checkpoint before exiting...")
        try:
            # Save checkpoint
            output_dir = trainer.args.output_dir
            checkpoint_dir = os.path.join(output_dir, "interrupt_checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainer.save_model(checkpoint_dir)
            
            # Save full model if requested
            save_model(model, tokenizer, "emergency_save", None)
            print(f"Emergency checkpoint saved to {checkpoint_dir}")
        except Exception as e:
            print(f"Error saving emergency checkpoint: {e}")
    
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 model with checkpoint recovery")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing data files")
    parser.add_argument("--batch_num", type=int, default=20, help="Number of batch files to load (0 to batch_num)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples (None = use all)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint directory to resume training from")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct", 
                        help="Base model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    
    # Checkpoint scheduling arguments
    parser.add_argument("--checkpoint_steps", type=int, default=500, 
                        help="Save recovery checkpoints every X steps (separate from eval checkpoints)")
    
    # Save arguments
    parser.add_argument("--model_name", type=str, default="llama1b_v1", help="Name to save the model as")
    parser.add_argument("--save_dir", type=str, default=None, help="Additional directory to save model (optional)")
    parser.add_argument("--save_on_interrupt", action="store_true", default=True, 
                        help="Save checkpoint on keyboard interrupt")
    
    return parser.parse_args()

from transformers import TrainerCallback
class CheckpointCallback(TrainerCallback):
    def __init__(self, trainer, checkpoint_steps, save_dir):
        self.trainer = trainer
        self.checkpoint_steps = checkpoint_steps 
        self.save_dir = save_dir
        self.last_checkpoint_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Required method that runs at start of training"""
        self.last_checkpoint_step = 0
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Save checkpoint at specified steps"""
        if state.global_step - self.last_checkpoint_step >= self.checkpoint_steps:
            checkpoint_dir = os.path.join(self.save_dir, f"recovery_step_{state.global_step}")
            self.trainer.save_model(checkpoint_dir)
            self.last_checkpoint_step = state.global_step
            print(f"\n--- Recovery checkpoint saved at step {state.global_step} ---\n")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Required method that runs at end of training"""
        return control

def main():
    global trainer, model, tokenizer, save_on_interrupt
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.max_split_size_mb = 512  # Limit CUDA memory splits
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    args = parse_args()
    save_on_interrupt = args.save_on_interrupt
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Free up any GPU memory that might be used
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load and prepare data
    print(f"\n=== Loading data from {args.data_dir} ===")
    combined_df = load_data(args.data_dir, args.batch_num)
    if combined_df is None:
        print("Data loading failed. Exiting.")
        return
    
    # Limit data samples if needed
    if args.max_samples and len(combined_df) > args.max_samples:
        print(f"Limiting dataset to {args.max_samples} samples (from {len(combined_df)})")
        combined_df = combined_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
    
    # Load model and tokenizer
    print(f"\n=== Loading model {args.base_model} ===")
    model, tokenizer = load_model(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )
    
    # Setup LoRA parameters
    print("\n=== Setting up LoRA parameters ===")
    model = setup_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth"  # Ensure this is enabled
    )
    
    # Split and prepare datasets
    print("\n=== Preparing datasets ===")
    train_df, val_df, test_df = split_data(combined_df)
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df, tokenizer)
    
    # Setup training arguments
    print("\n=== Setting up training configuration ===")
    training_args = get_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit
    )
    
    # Setup trainer with potential resumption from checkpoint
    resume_from = args.resume_from_checkpoint
    
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        compute_metrics=compute_metrics,
        training_args=training_args,
        max_seq_length=args.max_seq_length
    )
    
    # Add custom checkpoint callback
    checkpoint_callback = CheckpointCallback(
        trainer=trainer,
        checkpoint_steps=args.checkpoint_steps,
        save_dir=args.output_dir
    )
    trainer.add_callback(checkpoint_callback)
    
    # Start or resume training
    print("\n=== Starting training ===")
    try:
        trainer_stats = train_model(trainer, resume_from=resume_from)
        print("\n=== Training completed successfully ===")
    except Exception as e:
        print(f"\n!!! Training was interrupted by an exception: {e} !!!")
        print("Saving emergency checkpoint...")
        
        # Save emergency checkpoint
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        trainer.save_model(emergency_dir)
        print(f"Emergency checkpoint saved to {emergency_dir}")
        
        # Provide resumption instructions
        print("\nTo resume training, run the same command with:")
        print(f"  --resume_from_checkpoint {emergency_dir}")
        
        return
    
    # Free up GPU memory before saving
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save the final model
    print(f"\n=== Saving model as {args.model_name} ===")
    save_model(model, tokenizer, args.model_name, args.save_dir)
    
    print("\n=== Training pipeline complete! ===")

if __name__ == "__main__":
    main()