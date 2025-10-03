from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Paths
checkpoint = 4000
checkpoint_dir = f"/stt-whisper-training/models/whisper_model_default_tokenizer_on_wylie/checkpoint-{checkpoint}"

# Load model + processor
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
processor = WhisperProcessor.from_pretrained(
    checkpoint_dir, language="bo", task="transcribe"
)

model.generation_config.no_repeat_ngram_size = 2
model.generation_config.length_penalty = -1.0
model.generation_config.num_beams = 3   
# Your HF repo name (e.g., username/model_name)
repo_name = f"ganga4364/whisper_model_default_tokenizer_on_wylie/checkpoint-{checkpoint}"

# Push to hub
model.push_to_hub(repo_name)
processor.push_to_hub(repo_name)
