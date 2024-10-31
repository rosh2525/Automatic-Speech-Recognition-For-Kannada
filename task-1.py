from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from datasets import load_dataset

model_name = "facebook/wav2vec2-large-xlsr-53"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

dataset = load_dataset("path/to/sandalwood-dataset")

def preprocess(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids  # Convert transcription to tokens
    return batch

processed_dataset = dataset.map(preprocess, remove_columns=["audio", "text"])
