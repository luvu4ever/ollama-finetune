import os
import argparse
from src.data_utils import load_data, split_data, prepare_datasets
from src.model_utils import load_model, setup_peft_model, save_model, compute_metrics
from src.training_utils import get_training_args, setup_trainer, train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 model for text summarization")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing data files")
    parser.add_argument("--batch_num", type=int, default=20, help="Number of batch files to load (0 to batch_num)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use (for memory constraints)")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct", 
                        help="Base model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")  # REDUCED from 1024
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")  # REDUCED from 16
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device")  # REDUCED from 2
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")  # REDUCED from 16
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")  # REDUCED from 4
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X steps")  # INCREASED from 100
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every X steps")  # INCREASED from 100
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep")  # ADDED
    
    # Save arguments
    parser.add_argument("--model_name", type=str, default="llama1b_v1", help="Name to save the model as")
    parser.add_argument("--save_dir", type=str, default=None, 
                        help="Additional directory to save model (optional)")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Free up any GPU memory that might be used
    import torch
    torch.cuda.empty_cache()
    
    # Load and prepare data
    print(f"\n=== Loading data from {args.data_dir} ===")
    combined_df = load_data(args.data_dir, args.batch_num)
    if combined_df is None:
        print("Data loading failed. Exiting.")
        return
    
    # Limit data samples if needed
    if args.max_samples and len(combined_df) > args.max_samples:
        print(f"Limiting dataset to {args.max_samples} samples (from {len(combined_df)}) to reduce memory usage")
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
        save_total_limit=args.save_total_limit  # Added parameter
    )
    
    # Setup trainer
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        compute_metrics=compute_metrics,
        training_args=training_args,
        max_seq_length=args.max_seq_length
    )
    
    # Train model
    print("\n=== Starting training ===")
    trainer_stats = train_model(trainer)
    
    # Free up GPU memory before saving
    torch.cuda.empty_cache()
    
    # Save the model
    print(f"\n=== Saving model as {args.model_name} ===")
    save_model(model, tokenizer, args.model_name, args.save_dir)
    
    print("\n=== Training pipeline complete! ===")

if __name__ == "__main__":
    main()