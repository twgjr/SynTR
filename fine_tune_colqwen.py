import json
import os
import shutil
import torch
import warnings
from types import MethodType
from datasets import load_dataset
from transformers import TrainingArguments
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseNegativeCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.collators import CorpusQueryCollator
from colpali_engine.utils.gpu_stats import print_summary
from peft import LoraConfig
from transformers import BitsAndBytesConfig, EarlyStoppingCallback

try:
    import pynvml
except ImportError:
    warnings.warn("pynvml not found. GPU stats will not be printed.")

from vilarmor_dataset import ViLARMoRDataset
from evaluator import compute_metrics

def dataset_loading_func():
    data_files = {
        "train": "beir_splits/train.jsonl",
        "validation": "beir_splits/val.jsonl",
    }
    beir_dataset = load_dataset("json", data_files=data_files)

    # Instantiate ViLARMoRDataset to get the corpus images.
    # The dataset name should be the same as used when generating the splits.
    dataset_name = "vidore/docvqa_test_subsampled_beir"
    vil_dataset = ViLARMoRDataset(name=dataset_name, load_pseudos=True, load_judgements=False)
    corpus_dataset = vil_dataset.corpus

    # Specify the corpus format as used in your CorpusQueryCollator ("vidore" in this example)
    corpus_format = "vidore"

    # Return a tuple: (BEIR dataset splits, corpus dataset, corpus_format)
    return (beir_dataset, corpus_dataset, corpus_format)

class ColModelTrainingWithVal(ColModelTraining):
    def __init__(self, config, num_epochs):
        super().__init__(config)
        self.num_epochs = num_epochs

    def get_current_checkpoint_path(output_dir: str):
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(output_dir, name)
                return path
        return None

    def train(self) -> None:
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

        best_score = None
        patience = 2
        patience_counter = 0
        metric_key = "ndcg_at_10"

        prev_checkpoint_path=None
        for epoch in range(int(self.num_epochs)):
            print(f"\nStarting epoch {epoch + 1}")

            if prev_checkpoint_path is not None:
                print(f"Removing {prev_checkpoint_path}")
                shutil.rmtree(prev_checkpoint_path)

            trainer.train()

            print(f"Saving model to {self.config.tr_args.output_dir}")
            trainer.save_model(self.config.tr_args.output_dir)

            checkpoint_path = self.get_current_checkpoint_path(self.config.tr_args.output_dir)
            prev_checkpoint_path = checkpoint_path

            if checkpoint_path is None:
                print("No checkpoint folder found after saving. Skipping evaluation.")
                break

            print(f"Evaluating checkpoint at: {checkpoint_path}")
            metrics = compute_metrics(checkpoint_path=checkpoint_path, split_name="validation")

            current_score = metrics[metric_key]
            print(f"Current Score: {current_score}")

            if current_score is None:
                print(f"Metric '{metric_key}' not found in results.")
                break

            if best_score is None or current_score > best_score:
                best_score = current_score
                patience_counter = 0
                print(f"New best score {best_score:.4f}, saving as 'best'")
                trainer.save_model(f"{self.config.tr_args.output_dir}/best")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def config_model_training(best_checkpoint_dir:str=None):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16  
    )

    ##################################################################
    # Monkey-patch inner_forward to safely remove 'labels' if present
    from types import MethodType
    original_inner_forward = model.inner_forward

    def safe_inner_forward(self, *args, **kwargs):
        if "labels" in kwargs:
            kwargs.pop("labels")
        return original_inner_forward(*args, **kwargs)

    model.inner_forward = MethodType(safe_inner_forward, model)
    # End monkey patch
    #################################################################

    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False  
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
        resume_from_checkpoint=best_checkpoint_dir,
    )

    loss_func = ColbertPairwiseNegativeCELoss()

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

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
    return config

def main():
    checkpoint_dir="colqwen_beir_checkpoints/best"
    # config = config_model_training(checkpoint_dir)
    # training_app = ColModelTrainingWithVal(config, num_epochs=5)
    # training_app.train()

    # evaluate on test set
    compute_metrics(checkpoint_dir, split_name="test")

if __name__ == "__main__":
    main()