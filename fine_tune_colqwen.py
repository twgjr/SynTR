import json
import os
import shutil
import torch
import warnings
from tqdm import tqdm
from types import MethodType
from datasets import load_dataset
from transformers import TrainingArguments
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseNegativeCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from transformers import BitsAndBytesConfig, EarlyStoppingCallback, set_seed
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel



set_seed(42)  # for consistent testing, sets all seeds for randomness

try:
    import pynvml
except ImportError:
    warnings.warn("pynvml not found. GPU stats will not be printed.")

from vilarmor_dataset import ViLARMoRDataset
from evaluator import compute_metrics

def dataset_loading_func():
    data_files = {
        "train": "splits/general_judge-hard-3neg-1q/train.jsonl",
        "validation": "splits/general_judge-hard-3neg-1q/val.jsonl",
    }
    beir_dataset = load_dataset("json", data_files=data_files)

    # Instantiate ViLARMoRDataset to get the corpus images matching splits.
    dataset_name = "vidore/docvqa_test_subsampled_beir"
    vil_dataset = ViLARMoRDataset(
        name=dataset_name, 
        queries_path="pseudo_query_sets/general_judge-hard-3neg-1q/renum_filtered_pseudo_queries.json",
        qrels_path="pseudo_query_sets/general_judge-hard-3neg-1q/renum_filtered_pseudo_qrels.json")

    corpus_dataset = vil_dataset.corpus

    # Specify the corpus format as used in CorpusQueryCollator
    corpus_format = "vidore"

    # Return a tuple: (BEIR dataset splits, corpus dataset, corpus_format)
    return (beir_dataset, corpus_dataset, corpus_format)

class ColModelTrainingWithVal(ColModelTraining):
    def __init__(self, config):
        super().__init__(config)

    def train(self) -> None:
        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
            compute_metrics=None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.args.remove_unused_columns = False
        trainer.train()

def get_model_and_processor(checkpoint_dir: str = None, use_peft:bool=True):
    # Load Qwen‑base + Metric‑AI’s LoRA adapter in one go
    model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        device_map="auto",
        torch_dtype=torch.float16,
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

    if use_peft:
        if checkpoint_dir:
            # load LoRA trained and saved LoRA adapters
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            peft_config = None
        else:
            # Add new LoRA adapters yet to be trained.
            model = prepare_model_for_kbit_training(model)

            peft_config = LoraConfig(
                r=128,
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
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        peft_config = None

    # Load processor
    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False,
    )

    return model, processor, peft_config

def config_model_training(checkpoint_dir: str = None):
    model, processor, peft_config = get_model_and_processor(
        checkpoint_dir=checkpoint_dir,
        use_peft=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Training settings
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=20,
        bf16=True,
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir="./checkpoints",
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
        peft_config=None,  #handle peft outside of ColModelTrainingConfig
        pretrained_peft_model_name_or_path=None,
    )

    return config

def main():
    config = config_model_training(
        checkpoint_dir="results/general_judge-hard-3neg-1q/checkpoints/checkpoint-390"
    )
    training_app = ColModelTrainingWithVal(config)
    training_app.train()

if __name__ == "__main__":
    main()