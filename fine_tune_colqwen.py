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
from transformers import BitsAndBytesConfig, EarlyStoppingCallback, set_seed


set_seed(42)  # for consistent testing, sets all seeds for randomness

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
        patience = 3
        patience_counter = 0
        metric_key = "ndcg_at_10"

        for epoch in range(int(self.num_epochs)):
            print(f"\nStarting epoch {epoch + 1}")
            trainer.train()

            print(f"Validating")
            metrics = compute_metrics(
                checkpoint_path=checkpoint_path,
                split_name="validation",
                base_model_name="Metric-AI/colqwen2.5-3b-multilingual"
            )

            current_score = metrics[metric_key]
            print(f"Current Score: {current_score}")

            if current_score is None:
                print(f"Metric '{metric_key}' not found in results.")
                break

            if best_score is None or current_score > best_score:
                best_score = current_score
                patience_counter = 0
                print(f"New best score {best_score:.4f}, saving as 'best'")
                # trainer.save_model(f"{self.config.tr_args.output_dir}/best")
                self.model.save_pretrained(f"{self.config.tr_args.output_dir}/best")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def get_model_and_processor(checkpoint_dir: str = "colqwen_beir_checkpoints/best", use_peft:bool=True):
    from peft import PeftModel, get_peft_model, LoraConfig
    from types import MethodType

    # Load base model
    base_model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        device_map="auto",
        torch_dtype=torch.float16
    )

    if use_peft:
        if checkpoint_dir is None:
            # Define the new LoRA config for fresh starts from pretrained model
            new_peft_config = LoraConfig(
                r=32,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )

            new_peft_config.base_model_name_or_path = "Metric-AI/colqwen2.5-3b-multilingual"

            model = get_peft_model(base_model, new_peft_config)
        else:
            # Load PEFT adapter
            model = PeftModel.from_pretrained(model, checkpoint_dir)

        # Monkey-patch inner_forward
        original_inner_forward = model.inner_forward
        def safe_inner_forward(self, *args, **kwargs):
            if "labels" in kwargs:
                kwargs.pop("labels")
            return original_inner_forward(*args, **kwargs)
        model.inner_forward = MethodType(safe_inner_forward, model)

    # Load processor
    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False
    )

    return model, processor

def config_model_training():
    model, processor = get_model_and_processor(checkpoint_dir=None) 

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Training settings
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
    )

    loss_func = ColbertPairwiseNegativeCELoss()

    config = ColModelTrainingConfig(
        model=model,
        processor=processor,
        tr_args=training_args,
        dataset_loading_func=dataset_loading_func,
        run_eval=True,
        run_train=True,
        loss_func=loss_func,
        peft_config=new_peft_config,
    )

    return config

def main():
    config = config_model_training()
    training_app = ColModelTrainingWithVal(config, num_epochs=10)
    # training_app.train()

if __name__ == "__main__":
    from post_finetuning_eval import eval
    main()
    eval()