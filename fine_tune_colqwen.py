import torch
import warnings
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

try:
    import pynvml
except ImportError:
    warnings.warn("pynvml not found. GPU stats will not be printed.")

from vilarmor_dataset import ViLARMoRDataset

def dataset_loading_func():
    data_files = {
        "train": "beir_splits/train.jsonl",
        "validation": "beir_splits/val.jsonl",
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
        device_map="auto",
        torch_dtype=torch.float16  # ← helps reduce quantization warnings
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

    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False  # ← suppress tokenizer warning
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./colqwen_beir_checkpoints",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=9,
        bf16=True,
        save_steps=500,
        logging_steps=100,
        optim="paged_adamw_8bit",
        eval_strategy="epoch",  # ← replaces deprecated 'evaluation_strategy'
        eval_accumulation_steps=1,
        resume_from_checkpoint="./colqwen_beir_checkpoints/checkpoint-epoch6-loss2e-7",
    )

    loss_func = ColbertPairwiseNegativeCELoss()

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
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

    training_app = ColModelTrainingWithVal(config)
    training_app.train()

if __name__ == "__main__":
    main()
