import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_data(data_dir, batch_num):
    """
    Load and combine CSV files from the data directory.
    
    Args:
        data_dir (str): Path to directory containing CSV files
        batch_num (int): Number of batch files to load (0 to batch_num, inclusive)
        
    Returns:
        pandas.DataFrame or None: Combined dataframe if successful, None otherwise
    """
    all_dataframes = []
    
    for i in range(batch_num + 1):
        file_name = f"processed_batch_{i}.csv"
        file_path = os.path.join(data_dir, file_name)
        
        try:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
            print(f"Loaded {file_name}")
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    
    # Concatenate all dataframes into a single dataframe
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("\nSuccessfully loaded and combined all CSV files.")
        print(f"Total rows: {len(combined_df)}")
        print("Sample data:")
        print(combined_df.head())
        return combined_df
    else:
        print("\nNo CSV files were found or loaded successfully.")
        return None

def split_data(df):
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pandas.DataFrame): DataFrame to split
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
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

def prepare_datasets(train_df, val_df, test_df, tokenizer):
    """
    Convert dataframes to HuggingFace datasets.
    
    Args:
        train_df (pandas.DataFrame): Training data
        val_df (pandas.DataFrame): Validation data
        test_df (pandas.DataFrame): Test data
        tokenizer: Tokenizer for formatting prompts
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Apply formatting function with tokenizer provided
    format_with_tokenizer = lambda row: format_prompt(row, tokenizer)
    
    train_dataset = Dataset.from_pandas(train_df.apply(format_with_tokenizer, axis=1, result_type='expand'))
    val_dataset = Dataset.from_pandas(val_df.apply(format_with_tokenizer, axis=1, result_type='expand'))
    test_dataset = Dataset.from_pandas(test_df.apply(format_with_tokenizer, axis=1, result_type='expand'))
    
    print(f"Prepared datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset