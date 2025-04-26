import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import Dataset, concatenate_datasets
import gc
import torch

def load_data(data_dir, batch_num, max_length=2000):
    """
    Load and combine CSV files with improved error handling and validation.
    """
    all_dataframes = []
    total_rows = 0
    min_required_rows = 100  # Set minimum required samples
    
    for i in range(batch_num + 1):
        file_name = f"processed_batch_{i}.csv"
        file_path = os.path.join(data_dir, file_name)
        
        try:
            # Check if file exists and has content
            if not os.path.exists(file_path):
                print(f"File {file_name} not found. Skipping.")
                continue
                
            if os.path.getsize(file_path) == 0:
                print(f"File {file_name} is empty. Skipping.")
                continue
            
            # Read CSV with error handling
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if not all(col in df.columns for col in ["processed", "abstract"]):
                print(f"File {file_name} missing required columns. Skipping.")
                continue
            
            # Remove empty rows
            df = df.dropna(subset=["processed", "abstract"])
            
            # Filter out empty strings and whitespace-only entries
            df = df[df["processed"].str.strip().str.len() > 0]
            df = df[df["abstract"].str.strip().str.len() > 0]
            
            if len(df) > 0:
                all_dataframes.append(df)
                total_rows += len(df)
                print(f"Loaded {file_name}, valid rows: {len(df)}")
            else:
                print(f"No valid data in {file_name} after filtering. Skipping.")
                
        except pd.errors.EmptyDataError:
            print(f"File {file_name} is empty or malformed. Skipping.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    
    if not all_dataframes:
        print("\nNo valid data files were found.")
        return None
        
    if total_rows < min_required_rows:
        print(f"\nInsufficient data: Only found {total_rows} rows, minimum required is {min_required_rows}")
        return None
    
    # Combine valid dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nSuccessfully loaded {len(combined_df)} rows from {len(all_dataframes)} files")
    print("Sample data:")
    print(combined_df.head())
    
    return combined_df

def split_data(df, val_size=0.15, test_size=0.15):
    """
    Split data into train, validation, and test sets with smaller validation/test sets.
    
    Args:
        df (pandas.DataFrame): DataFrame to split
        val_size (float): Portion for validation 
        test_size (float): Portion for testing
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Calculate test_size relative to entire dataset
    test_fraction = test_size
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(df, test_size=test_fraction, random_state=42)
    
    # Second split: separate validation set from training set
    # val_size relative to what remains after test set is removed
    val_fraction = val_size / (1 - test_fraction)
    train_df, val_df = train_test_split(train_val_df, test_size=val_fraction, random_state=42)
    
    # Free memory
    del train_val_df
    gc.collect()
    
    print("Train data shape:", train_df.shape)
    print("Validation data shape:", val_df.shape)
    print("Test data shape:", test_df.shape)
    
    return train_df, val_df, test_df

def format_prompt(row, tokenizer):
    """
    Format a row of data into the chat template format.
    
    Args:
        row (pandas.Series): Row of data
        tokenizer: Tokenizer for applying chat template
        
    Returns:
        dict: Formatted data
    """
    prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes scientific text into concise and clear summaries."
        },
        {
            "role": "user",
            "content": row["processed"]
        },
        {
            "role": "assistant",
            "content": row["abstract"]
        }
    ]
    return {
        "text": tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False),
        "summary": row["abstract"]
    }

def prepare_datasets(train_df, val_df, test_df, tokenizer, max_samples=None):
    """
    Convert dataframes to HuggingFace datasets.
    
    Args:
        train_df (pandas.DataFrame): Training data
        val_df (pandas.DataFrame): Validation data
        test_df (pandas.DataFrame): Test data
        tokenizer: Tokenizer for formatting prompts
        max_samples (int, optional): Maximum number of samples per dataset
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Limit dataset size if needed
    # if max_samples:
    #     if len(train_df) > max_samples:
    #         train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
    #     val_size = max(50, int(max_samples * 0.1))  # At least 50 validation samples
    #     if len(val_df) > val_size:
    #         val_df = val_df.sample(n=val_size, random_state=42).reset_index(drop=True)
            
    #     test_size = max(50, int(max_samples * 0.1))  # At least 50 test samples
    #     if len(test_df) > test_size:
    #         test_df = test_df.sample(n=test_size, random_state=42).reset_index(drop=True)
    
    # Apply formatting function with tokenizer provided
    format_with_tokenizer = lambda row: format_prompt(row, tokenizer)
    print("Processing training data sequentially...")
    # Process in smaller chunks to save memory
    train_chunks = []
    chunk_size = 1000
    
    for i in range(0, len(train_df), chunk_size):
        try:
            chunk_df = train_df.iloc[i:i+chunk_size]
            chunk_dataset = Dataset.from_pandas(chunk_df.apply(format_with_tokenizer, axis=1, result_type='expand'))
            train_chunks.append(chunk_dataset)
            
            print(f"Processed chunk {i//chunk_size + 1}/{(len(train_df)-1)//chunk_size + 1}")
            
            # Free memory
            del chunk_df
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing chunk {i//chunk_size + 1}: {e}")
            continue
    
    # Combine chunks using concatenate_datasets
    try:
        if len(train_chunks) > 0:
            train_dataset = concatenate_datasets(train_chunks)
            print(f"Successfully combined {len(train_chunks)} chunks")
        else:
            raise ValueError("No valid training chunks to combine")
            
    except Exception as e:
        print(f"Error combining chunks: {e}")
        return None, None, None
    
    # Free memory
    del train_chunks
    gc.collect()
    
    # Process validation and test data
    try:
        val_dataset = Dataset.from_pandas(val_df.apply(format_with_tokenizer, axis=1, result_type='expand'))
        test_dataset = Dataset.from_pandas(test_df.apply(format_with_tokenizer, axis=1, result_type='expand'))
    except Exception as e:
        print(f"Error processing validation/test data: {e}")
        return None, None, None
    
    print(f"Final dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset