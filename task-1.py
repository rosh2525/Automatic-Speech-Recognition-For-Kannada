from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

model_name = "facebook/wav2vec2-large-xlsr-53"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

dataset = load_dataset("Sandalwood")

def preprocess(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids  # Convert transcription to tokens
    return batch

processed_dataset = dataset.map(preprocess, remove_columns=["audio", "text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-kannada",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    weight_decay=0.005,
    logging_steps=10,
    save_steps=100,
    num_train_epochs=10,
    save_total_limit=2,
    fp16=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    data_collator=lambda data: {"input_values": torch.tensor([f["input_values"] for f in data]),
                                "labels": torch.tensor([f["labels"] for f in data])},
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
)

# Fine-tune the model
trainer.train()
