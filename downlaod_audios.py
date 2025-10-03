import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import requests
import pandas as pd
    from datasets import load_dataset
import os
import requests
import logging

# Configure logging
logging.basicConfig(filename='failed_downloads.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Function to download audio file
def download_audio(url, file_name, save_path):
    # Full path where the audio will be saved
    
    # Skip download if file already exists
    if os.path.exists(save_path):
        return
    
    try:
        # Download the file from the URL and save it to the specified path
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        # Log the error and file name if the download fails
        logging.error(f"Failed to download {file_name}: {e}")


# Function to download the audio file and perform inference
def process_row(row):
    url = row['url']
    file_name = os.path.basename(url)
    
    # Create the save path for the temporary audio file
    save_path = f"/stt-whisper-training/data/wav_16k/{file_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Function to download the audio file
    download_audio(url, file_name, save_path)
    
    return row

# Function to process the DataFrame using Pool
def process_dataframe(df):
    # Number of processes to run, set to the number of available CPU cores or adjust as needed
    num_workers = min(cpu_count(), 5)  # Adjust based on your system capabilities

    # Use Pool to process rows in parallel
    with Pool(processes=num_workers) as pool:
        rows = list(tqdm(pool.imap(process_row, [row for _, row in df.iterrows()]), total=len(df)))

    # Convert the processed rows back into a DataFrame
    return pd.DataFrame(rows)

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("openpecha/garchen_rinpoche_data")

    # Convert train, validation, and test splits to DataFrames
    train_df = dataset["train"].to_pandas()
    validation_df = dataset["validation"].to_pandas()

    # Add a column to identify the split
    train_df["split"] = "train"
    validation_df["split"] = "validation"

    # Concatenate all splits into a single DataFrame
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)

    # Save to a CSV file if needed
    combined_df.to_csv("combined_dataset.csv", index=False)
    print("Combined DataFrame saved to 'combined_dataset.csv'")

    # Process the DataFrame
    df = process_dataframe(combined_df)

    # Save the processed DataFrame to a CSV file
    df.to_csv('processed_data.csv', index=False)
    print("Processed DataFrame saved to 'processed_data.csv'")
