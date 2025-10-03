import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ganga4364/garchen_rinpoche_inference_transcripts")

test_df = dataset["train"].to_pandas()
test_df.to_csv('test.csv', index=False)

import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import logging
import requests
import torch
import torchaudio
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pandarallel import pandarallel
import pandas as pd
from datasets import load_dataset



# -------------------------------
# Config
# -------------------------------

INPUT_FILE = "/stt-whisper-training/test.csv"   # CSV must have a "url" column
OUTPUT_DIR = "/stt-whisper-training/whisper_model_default_tokenizer_on_wylie/output_chunks"
DOWNLOAD_DIR = "/stt-whisper-training/downloads"
CHUNK_SIZE = 1000         # how many rows to load at once from CSV
BATCH_SIZE = 4            # how many rows to process per map() batch
NUM_WORKERS = 4
TIMEOUT = 20
OUTPUT_FILE = "output.csv"


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Load the dataset
dataset = load_dataset("ganga4364/garchen_rinpoche_inference_transcripts")

test_df = dataset["train"].to_pandas()
test_df.to_csv(INPUT_FILE, index=False)


logging.basicConfig(filename="failed_downloads.log",
                    level=logging.ERROR,
                    format="%(asctime)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load both models + processors
# -------------------------------
CHECKPOINT = "ganga4364/whisper_model_default_tokenizer_on_wylie/checkpoint-4000"

processor = WhisperProcessor.from_pretrained(
    CHECKPOINT, language="bo", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)

# -------------------------------
# Helper functions
# -------------------------------
import pandas as pd
import pyewts

# Initialize converter
converter = pyewts.pyewts()

def download_audio(row):
    """Download file from row['url'] → save to local dir."""
    try:
        file_name = os.path.basename(row["url"])
        save_path = os.path.join(DOWNLOAD_DIR, file_name)

        if os.path.exists(save_path):
            return save_path

        response = requests.get(row["url"], timeout=TIMEOUT)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return save_path
    except Exception as e:
        logging.error(f"Failed to download {row.get('file_name', row['url'])}: {e}")
        return None


def transcribe_batch(audio_paths, model, processor):
    """Run Whisper inference on a batch of audio files using GPU efficiently."""
    waveforms = []
    sample_rates = []

    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        waveforms.append(waveform.squeeze().numpy())   # ✅ ensure numpy array
        sample_rates.append(sr)

    # Resample individually if needed
    target_sr = 16000
    if any(sr != target_sr for sr in sample_rates):
        resampled = []
        for wf, sr in zip(waveforms, sample_rates):
            if sr != target_sr:
                wf_t = torch.tensor(wf)
                wf_t = torchaudio.transforms.Resample(sr, target_sr)(wf_t)
                resampled.append(wf_t.numpy())
            else:
                resampled.append(wf)
        waveforms = resampled

    # Processor handles padding automatically
    inputs = processor(
        waveforms,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    # Generate in batch
    pred_ids = model.generate(
        inputs["input_features"],
        num_beams=4,
        max_length=225,
        length_penalty=1.0,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return processor.batch_decode(pred_ids, skip_special_tokens=True)


def process_inference(batch):
    """Run both Tibetan + Wylie models on a batch of files (batched on GPU)."""
    batch[f"{CHECKPOINT}_transcript"] = transcribe_batch(batch["path"], model, processor)
    return batch

# -------------------------------
# Parallel init
# -------------------------------
pandarallel.initialize(progress_bar=True, nb_workers=NUM_WORKERS)

# -------------------------------
# Main loop
# -------------------------------
for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE), start=1):
    if chunk.empty:
        print(f"⚠️ Chunk {i} is empty. Skipping.")
        continue

    chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{i}.csv")
    if os.path.exists(chunk_file):
        print(f"⏩ Skipping chunk {i}, already processed.")
        continue

    # Download audio in parallel
    tqdm.pandas(desc=f"Downloading chunk {i}")
    chunk["path"] = chunk.parallel_apply(download_audio, axis=1)

    # Drop failed downloads
    chunk = chunk[chunk["path"].notnull()].reset_index(drop=True)
    if chunk.empty:
        print(f"⚠️ All downloads failed for chunk {i}. Skipping.")
        continue

    # Convert to HF Dataset and run inference
    dataset = Dataset.from_pandas(chunk)
    dataset = dataset.map(process_inference, batched=True, batch_size=BATCH_SIZE)

    # Save to CSV
    dataset.to_pandas().to_csv(chunk_file, index=False)
    print(f"✅ Saved chunk {i} → {chunk_file}")

print("🎉 All chunks processed!")

# Merge all chunk files into final output
all_chunks = [pd.read_csv(os.path.join(OUTPUT_DIR, f)) for f in sorted(os.listdir(OUTPUT_DIR)) if f.endswith(".csv")]
final_df = pd.concat(all_chunks, ignore_index=True)
# Convert to Tibetan
#final_df[f"{CHECKPOINT}_transcript_wylie_to_tibetan"] = final_df[f"{CHECKPOINT}_transcript"].apply(converter.toUnicode)
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"All chunks merged and saved to {OUTPUT_FILE}")