import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import unsloth_train

def get_training_args(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=5,
    learning_rate=2e-4,
    num_train_epochs=4,
    save_steps=100,
    eval_steps=100
):
    return TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",  # "none" for console logs; use "tensorboard" or "wandb" for visual logging
        
        logging_steps=10,
        logging_strategy="steps",
        
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        
        load_best_model_at_end=True,
        save_only_model=False
    )

def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    compute_metrics,
    training_args,
    max_seq_length=1024
):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",  # Full chat-formatted prompt
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        compute_metrics=compute_metrics,
        args=training_args
    )
    
    return trainer

def train_model(trainer):
    """
    Train the model using unsloth optimizations.
    
    Args:
        trainer: Configured trainer
        
    Returns:
        dict: Training statistics
    """
    print("Starting training...")
    trainer_stats = unsloth_train(trainer)
    print("Training complete!")
    
    return trainer_stats