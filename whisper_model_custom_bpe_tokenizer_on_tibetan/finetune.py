import torch
import wandb
import functools
from datasets import load_from_disk
from transformers import (
    WhisperConfig,
    WhisperTokenizerFast,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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

# Initialize WandB
wandb.login(key='cd3943d98b7ac4e0a0abd24721f4016a7942166f')
wandb.init(project="stt-for-tibetan-language", entity="stt-for-tibet")

# Load components
print("Loading tokenizer and model...")

# IMPROVED APPROACH: Load pre-trained model first, then adapt
print("Loading pre-trained Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load custom tokenizer
tokenizer = WhisperTokenizerFast.from_pretrained(
    "/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan_processor",
    language="bo",
    task="transcribe"
)

# Load feature extractor (keep the original one)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
processor = WhisperProcessor(feature_extractor, tokenizer)

# CRITICAL: Resize token embeddings to match new vocabulary
print(f"Original vocab size: {model.config.vocab_size}")
print(f"New vocab size: {len(tokenizer)}")

if len(tokenizer) != model.config.vocab_size:
    print("Resizing token embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    
    # Update config
    model.config.vocab_size = len(tokenizer)

# Update special token IDs in config
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Configure the model for Tibetan language
print("Configuring model for custom tokenizer and Tibetan language...")

# Get Tibetan language decoder prompt IDs
forced_decoder_ids = processor.get_decoder_prompt_ids(language="bo", task="transcribe")

# Create custom generation config
custom_gen_config = GenerationConfig.from_model_config(model.config)
custom_gen_config.forced_decoder_ids = forced_decoder_ids
custom_gen_config.language = "bo"
custom_gen_config.task = "transcribe"

# Add missing attributes that Whisper's generate method expects
bo_token_id = tokenizer.convert_tokens_to_ids("<|bo|>")
transcribe_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")

# Handle case where tokens might not exist
if bo_token_id is not None:
    custom_gen_config.lang_to_id = {"bo": bo_token_id}
if transcribe_token_id is not None:
    custom_gen_config.task_to_id = {"transcribe": transcribe_token_id}

model.generation_config = custom_gen_config

# Monkey patch model.generate to always use our custom generation config
original_generate = model.generate
def patched_generate(self, *args, **kwargs):
    if "generation_config" not in kwargs:
        kwargs["generation_config"] = custom_gen_config
    return original_generate(*args, **kwargs)

model.generate = functools.partial(patched_generate, model)

# Set use_cache to False for gradient checkpointing compatibility
model.config.use_cache = False

# Load datasets
print("Loading datasets...")
train_dataset = load_from_disk('/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/train')
val_dataset = load_from_disk('/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/validation')

# Create a smaller validation subset for efficiency
val_subset = val_dataset.select(range(min(100, len(val_dataset))))

# Data collator
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

# Metrics function
def compute_metrics(pred):
    from tibetan_wer.metrics import wer
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    print("Sample prediction: ", pred_str)
    print("Sample reference: ", label_str)
    
    # Calculate WER
    wer_result = wer(predictions=pred_str, references=label_str)
    return {"wer": wer_result['micro_wer']}

# Training arguments - ADJUSTED for fine-tuning instead of training from scratch
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tibetan-finetuned",
    
    # Training parameters - REDUCED for fine-tuning
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,     # LOWER learning rate for fine-tuning
    warmup_steps=500,       # FEWER warmup steps for fine-tuning
    max_steps=15000,         # FEWER steps for fine-tuning
    
    # Settings to avoid CUDA errors
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,
    
    # Evaluation settings
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    
    # Checkpoint settings
    save_steps=1000,         # More frequent saves for fine-tuning
    eval_steps=1000,         # More frequent evaluation
    logging_steps=25,
    
    # Reporting and metrics
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    # Additional settings
    dataloader_num_workers=4,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    save_strategy="steps",
    max_grad_norm=1.0,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# Create trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_subset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and save
if __name__ == "__main__":
    check_gpu()
    
    # Clear any cached gradients
    model.zero_grad()
    
    print(f"Starting fine-tuning with {len(train_dataset)} examples")
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    print("ðŸš€ Starting training...")
    try:
        trainer.train()
        
        # Save everything
        print("ðŸ’¾ Saving model...")
        trainer.save_model("/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/final")
        processor.save_pretrained("/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/final")
        custom_gen_config.save_pretrained("/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/final")
        
        print("âœ… Training completed successfully!")
        
    except Exception as e:
        print(f"âŒ Training failed: {e}")
        # Still try to save checkpoint
        try:
            trainer.save_model("/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/final-checkpoint")
            processor.save_pretrained("/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/final-checkpoint")
            custom_gen_config.save_pretrained("/stt-whisper-training/data/whisper_model_custom_bpe_tokenizer_on_tibetan/final-checkpoint")
            print("ðŸ’¾ Checkpoint saved despite training failure")
        except Exception as save_error:
            print(f"âŒ Could not save checkpoint: {save_error}")
    
    finally:
        wandb.finish()