import torch
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk
from transformers import (
    WhisperTokenizerFast,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# Remove duplicate imports and fix WandB initialization
wandb.login(key='cd3943d98b7ac4e0a0abd24721f4016a7942166f')

wandb.init(
    project="stt-for-tibetan-language",
    entity="stt-for-tibet"
)

# --- 1. GPU Check ---
def check_gpu():
    """Check for GPU availability."""
    try:
        if not torch.cuda.is_available():
            print('Not connected to a GPU')
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    except Exception as e:
        print(f"An error occurred while checking for GPU: {e}")

# --- 2. Load Tokenizer, Feature Extractor, and Processor ---
print("Loading tokenizer, feature extractor, and processor...")


tokenizer = WhisperTokenizerFast.from_pretrained(
    "/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan_processor",
    language="bo",  # âœ… Correct ISO language code for Tibetan
    task="transcribe"
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small"
)

processor = WhisperProcessor(
    feature_extractor=feature_extractor, 
    tokenizer=tokenizer
)

# --- 3. Load Model ---
print("Loading pre-trained Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.resize_token_embeddings(len(tokenizer))


#model.generation_config.forced_decoder_ids = None
#model.generation_config.no_repeat_ngram_size = 2
#model.config.dropout = 0.1
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="bo", task="transcribe")

# CRITICAL FIX: Set use_cache to False for gradient checkpointing compatibility
model.config.use_cache = False

# --- 4. Load Datasets ---
print("Loading datasets...")
dataset_train = load_from_disk('/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan/train')
dataset_val = load_from_disk('/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan/validation')

val_subset = dataset_val.select(range(min(50, len(dataset_val))))

# --- 5. Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and (labels[:, 0] == bos_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# --- 6. Metrics ---
from tibetan_wer.metrics import wer

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # We do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    print("pred_str: ", pred_str)
    print("label_str: ", label_str)

    wer_result = wer(predictions=pred_str, references=label_str)
    return {"wer": wer_result['micro_wer']}

# --- 7. Training Arguments (Fixed Configuration) ---
print("Setting up training arguments...")

training_args = Seq2SeqTrainingArguments(
    output_dir="/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan",
    per_device_train_batch_size=16,  # Reduced batch size
    gradient_accumulation_steps=1,  # Increased to maintain effective batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=8000,
    
    # Option 1: Disable gradient checkpointing (RECOMMENDED)
    gradient_checkpointing=True,
    fp16=False,                    # ðŸ”¥ DISABLED for compatibility
    bf16=False,
    # Option 2: If you need gradient checkpointing, use these settings instead:
    #gradient_checkpointing=True,
    #fp16=False,  # Disable fp16 when using gradient checkpointing
    #bf16=False,  # Also disable bf16
    
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,  # More frequent saves
    eval_steps=100,  # More frequent evaluation
    logging_steps=25,
    report_to=["wandb"],  # Changed back to wandb
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    
    # Additional fixes
    dataloader_num_workers=4,  # Set to 0 to avoid multiprocessing issues
    remove_unused_columns=False,  # Important for speech data
    
    # Memory optimization
    dataloader_pin_memory=False,
    save_strategy="steps",
    
    # Prevent gradient issues
    max_grad_norm=1.0,  # Gradient clipping,
    gradient_checkpointing_kwargs={"use_reentrant": False},

)

# --- 8. Trainer ---
print("Initializing trainer...")

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_train,
    eval_dataset=val_subset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,  # Fixed: use processing_class instead of tokenizer
)

# --- 9. Training ---
if __name__ == "__main__":
    check_gpu()
    
    # Clear any cached gradients
    model.zero_grad()
    
    print("Starting training...")
    try:
        trainer.train()
        print("Training finished successfully.")
        
        # --- 10. Save Model ---
        print("Saving model...")
        trainer.save_model("/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan/final")
        processor.save_pretrained("/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan/final")
        print("Model saved.")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save checkpoint even if training fails
        trainer.save_model("/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan/final-checkpoint")
        processor.save_pretrained("/stt-whisper-training/data/whisper_model_added_tibetan_tokens_tokenizer_on_tibetan/final-checkpoint")
        raise e
    
    finally:
        wandb.finish()