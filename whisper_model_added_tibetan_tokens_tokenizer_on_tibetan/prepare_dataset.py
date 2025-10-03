import os
import torchaudio
from datasets import load_dataset
from transformers import WhisperProcessor

# -------------------------------
# 1. Load your custom processor
# -------------------------------
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
from transformers import PreTrainedTokenizerFast 

from transformers import PreTrainedTokenizerFast 

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan_processor"
)
# keep using the pretrained Whisper feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


# combine into processor
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",   # still pulls feature_extractor + config
)
# overwrite tokenizer inside the processor with yours
processor.tokenizer = tokenizer

print("âœ… Custom tokenizer loaded with vocab size:", tokenizer.vocab_size)

# -------------------------------
# 2. Load dataset from Hugging Face Hub
# -------------------------------
# Example: dataset has 'file_name' for each audio + 'uni' for Tibetan transcript
dataset = load_dataset("openpecha/garchen_rinpoche_data")

# Add full audio path column (pointing to local wav files you prepared)
dataset = dataset.map(
    lambda x: {"path": f"/stt-whisper-training/data/wav_16k/{x['file_name']}.wav"}
)

# -------------------------------
# 3. Preprocessing function
# -------------------------------

def prepare_dataset(batch):
    # load audio manually
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    if sampling_rate != processor.feature_extractor.sampling_rate:
        speech_array = torchaudio.functional.resample(speech_array, sampling_rate, processor.feature_extractor.sampling_rate)

    # Whisper expects mono
    speech_array = speech_array.squeeze().numpy()   

    # extract log-mel spectrogram
    inputs = processor.feature_extractor(
        speech_array, 
        sampling_rate=processor.feature_extractor.sampling_rate, 
        return_tensors="pt"
    )

    batch["input_features"] = inputs.input_features[0]
    
    # encode target text to label ids with truncation
    # Whisper's max sequence length is 1024, but we use a conservative value
   # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["uni"]).input_ids
    return batch

# -------------------------------
# 4. Apply preprocessing (debug mode - first 10 rows only)
# -------------------------------
print("ðŸ” DEBUG MODE: Processing only first 10 rows of each dataset")

# Select just first 10 rows for debugging
#train_subset = dataset["train"].select(range(10))
#val_subset = dataset["validation"].select(range(min(10, len(dataset["validation"]))))
train_subset = dataset["train"]
val_subset = dataset["validation"]
train_dataset = train_subset.map(
    prepare_dataset,
    remove_columns=train_subset.column_names,
    desc="Preparing train dataset (10 samples)",
)

val_dataset = val_subset.map(
    prepare_dataset,
    remove_columns=val_subset.column_names,
    desc="Preparing validation dataset (10 samples)",
)

# -------------------------------
# 5. Save preprocessed datasets
# -------------------------------
print("val_dataset[labels]: ", val_dataset["labels"][0])
print("train_dataset[labels]: ", train_dataset["labels"][0])
output_dir = "/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan"
train_dataset.save_to_disk(os.path.join(output_dir, "train"))
val_dataset.save_to_disk(os.path.join(output_dir, "validation"))

print("âœ… Preprocessed train/val datasets saved to", output_dir)