import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseNegativeCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.collators import CorpusQueryCollator
from colpali_engine.utils.gpu_stats import print_summary
from peft import LoraConfig
from transformers import BitsAndBytesConfig

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
    vil_dataset = ViLARMoRDataset(name=dataset_name, load_pseudos=True, load_judgements=True)
    corpus_dataset = vil_dataset.corpus

    # Specify the corpus format as used in your CorpusQueryCollator ("vidore" in this example)
    corpus_format = "vidore"

    # Return a tuple: (BEIR dataset splits, corpus dataset, corpus_format)
    return (beir_dataset, corpus_dataset, corpus_format)

class ColModelTrainingWithVal(ColModelTraining):
    def train(self) -> None:
        if isinstance(self.collator, CorpusQueryCollator) and self.collator.mined_negatives:
            print("Training with hard negatives")
        else:
            print("Training with in-batch negatives")

        # IMPORTANT: use dataset["validation"] for eval_dataset
        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
        )

        trainer.args.remove_unused_columns = False
        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

def main():
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        quantization_config=bnb_config,
        device_map="auto"
    )

    from types import MethodType

    ##################################################################
    # Monkey-patch inner_forward to safely remove 'labels' if present
    original_inner_forward = model.inner_forward

    def safe_inner_forward(self, *args, **kwargs):
        if "labels" in kwargs:
            kwargs.pop("labels")
        return original_inner_forward(*args, **kwargs)

    model.inner_forward = MethodType(safe_inner_forward, model)
    # End monkey patch
    #################################################################
    
    processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")

    # Define training arguments using transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir="./colqwen_beir_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        save_steps=500,
        logging_steps=100,
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",
    )

    loss_func = ColbertPairwiseNegativeCELoss()


    peft_config = LoraConfig(
        r=32,  # Reduced from 128
        lora_alpha=32,  # Reduced proportionally
        target_modules=["q_proj", "v_proj"],  # Focusing on attention modules
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",  # Adjust based on your specific task
    )

    # Create a training configuration using ColModelTrainingConfig.
    # This config uses the dataset_loading_func to load the BEIR splits and corpus images.
    config = ColModelTrainingConfig(
        model=model,
        processor=processor,
        tr_args=training_args,
        dataset_loading_func=dataset_loading_func,
        run_eval=True,
        run_train=True,
        loss_func=loss_func,
        peft_config=peft_config,
    )


    # Initialize the training application and train the model.
    training_app = ColModelTrainingWithVal(config)
    training_app.train()

    # Save the finetuned model and training configuration.
    training_app.save("finetune_beir_colqwen_config.yml")

if __name__ == "__main__":
    main()
