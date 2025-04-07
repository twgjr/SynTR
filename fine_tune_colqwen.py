import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig

# Import your VilarmorDataset to load the corpus (images)
from vilarmor_dataset import ViLARMoRDataset

def dataset_loading_func():
    """
    Loads the BEIR-style splits and returns a tuple:
      (dataset_splits, corpus_dataset, corpus_format)
    The dataset_splits is a dict with keys "train", "validation", "test" loaded from the JSONL files.
    The corpus_dataset is loaded from ViLARMoRDataset and contains the image data.
    """
    # Load BEIR splits JSONL files (adjust paths if needed)
    data_files = {
        "train": "beir_splits/train.jsonl",
        "validation": "beir_splits/val.jsonl",
        "test": "beir_splits/test.jsonl"
    }
    beir_dataset = load_dataset("json", data_files=data_files)

    # Instantiate ViLARMoRDataset to get the corpus images.
    # The dataset name should be the same as used when generating the splits.
    dataset_name = "vidore/docvqa_test_subsampled_beir"
    vil_dataset = ViLARMoRDataset(name=dataset_name, load_pseudos=False, load_judgements=False)
    corpus_dataset = vil_dataset.corpus

    # Specify the corpus format as used in your CorpusQueryCollator ("vidore" in this example)
    corpus_format = "vidore"

    # Return a tuple: (BEIR dataset splits, corpus dataset, corpus_format)
    return (beir_dataset, corpus_dataset, corpus_format)

def main():
    # Load the model and processor
    model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")

    # Define training arguments using transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir="./colqwen_beir_checkpoints",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        save_steps=500,
        logging_steps=100,
        # Additional TrainingArguments parameters can be set as needed.
    )

    # Create a training configuration using ColModelTrainingConfig.
    # This config uses the dataset_loading_func to load the BEIR splits and corpus images.
    config = ColModelTrainingConfig(
        model=model,
        processor=processor,
        tr_args=training_args,
        dataset_loading_func=dataset_loading_func,
        run_eval=True,
        run_train=True
    )

    # Initialize the training application and train the model.
    training_app = ColModelTraining(config)
    training_app.train()

    # Save the finetuned model and training configuration.
    training_app.save("finetune_beir_colqwen_config.yml")

if __name__ == "__main__":
    main()
