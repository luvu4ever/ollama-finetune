import torch
import shutil
from unsloth import FastLanguageModel

def load_model(model_name="unsloth/Llama-3.2-1B-Instruct", max_seq_length=1024, dtype=None, load_in_4bit=True):
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    print("Model loaded successfully")
    
    return model, tokenizer

def setup_peft_model(
    model, 
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth"
):
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
    
    return model

def save_model(model, tokenizer, model_name, destination_dir=None):
    # Save locally
    print(f"Saving model to {model_name}")
    model.save_pretrained(model_name)
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
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {k: round(v, 4) for k, v in result.items()}