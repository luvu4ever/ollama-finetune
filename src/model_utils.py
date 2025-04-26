import torch
import shutil
import gc
from unsloth import FastLanguageModel

def load_model(model_name="unsloth/Llama-3.2-1B-Instruct", max_seq_length=512, dtype=None, load_in_4bit=True):
    """Load model with better error handling"""
    try:
        # Clear GPU memory before loading
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check CUDA availability and memory
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            print(f"Available CUDA memory: {free_mem / 1024**2:.2f}MB")
        
        # Load model with error catching
        print(f"Loading model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            torch_dtype=torch.float16,  # Force fp16
            low_cpu_mem_usage=True
        )
        
        return model, tokenizer
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error during model loading: {e}")
            print("Try reducing max_seq_length or using load_in_4bit=True")
        raise

def setup_peft_model(
    model, 
    r=8,  # REDUCED from 16
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth"
):
    """
    Set up LoRA parameters for efficient fine-tuning.
    
    Args:
        model: Base model
        r (int): LoRA rank
        target_modules (list): Modules to apply LoRA to
        lora_alpha (int): LoRA alpha value
        use_gradient_checkpointing (str or bool): Gradient checkpointing strategy
        
    Returns:
        model: Model with LoRA configuration
    """
    print(f"Setting up PEFT model with r={r}, lora_alpha={lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Optimized setting
        bias="none",     # Optimized setting
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    print("PEFT model setup complete")
    
    # Free unused memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return model

def save_model(model, tokenizer, model_name, destination_dir=None):
    """
    Save model and tokenizer to specified locations.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        model_name (str): Directory name to save model
        destination_dir (str, optional): Additional location to copy saved model
    """
    # Clear cache before saving
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save locally
    print(f"Saving model to {model_name}")
    model.save_pretrained(model_name, safe_serialization=True)  # Use safe serialization
    tokenizer.save_pretrained(model_name)
    print(f"Model saved locally to {model_name}")
    
    # Save to destination directory if specified
    if destination_dir:
        try:
            print(f"Copying model to {destination_dir}")
            shutil.copytree(model_name, destination_dir)
            print(f"Successfully copied '{model_name}' to '{destination_dir}'")
        except FileExistsError:
            print(f"Directory '{destination_dir}' already exists. Skipping copy.")
        except Exception as e:
            print(f"An error occurred while copying: {e}")

def compute_metrics(eval_preds):
    """
    Compute evaluation metrics (ROUGE) for model predictions.
    
    Args:
        eval_preds: Tuple of (predictions, labels)
        
    Returns:
        dict: Metrics dictionary
    """
    import evaluate
    rouge = evaluate.load("rouge")
    preds, labels = eval_preds
    
    # Get the tokenizer from the global scope
    # This is a bit of a hack, but it's a common pattern in HF's compute_metrics functions
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    
    # Process in smaller batches to save memory
    batch_size = 8
    decoded_preds = []
    decoded_labels = []
    
    for i in range(0, len(preds), batch_size):
        batch_preds = preds[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        decoded_preds.extend(tokenizer.batch_decode(batch_preds, skip_special_tokens=True))
        decoded_labels.extend(tokenizer.batch_decode(batch_labels, skip_special_tokens=True))
        
        # Free memory
        del batch_preds, batch_labels
        torch.cuda.empty_cache()
    
    # Clean up whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {k: round(v, 4) for k, v in result.items()}