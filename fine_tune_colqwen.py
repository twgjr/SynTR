# fine_tune_colqwen.py
import torch
from datasets import load_dataset
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.training import ColTrainer, ColTrainingArguments
from colpali_engine.data import QueryDocumentDataset

# Load model & processor
model = ColQwen2_5.from_pretrained(
    "Metric-AI/colqwen2.5-3b-multilingual",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")

# Load dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl"})["train"]
train_dataset = QueryDocumentDataset(dataset, processor)

# Define training args
training_args = ColTrainingArguments(
    output_dir="./colqwen-checkpoints",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    save_steps=500,
    logging_steps=100,
    use_lora=True,
    lora_r=128,
    lora_alpha=128,
    optim="paged_adamw_8bit",
)

# Trainer
trainer = ColTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
trainer.train()
