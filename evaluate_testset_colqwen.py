import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, BitsAndBytesConfig
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.collators import CorpusQueryCollator
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseNegativeCELoss
from peft import PeftModel
from vilarmor_dataset import ViLARMoRDataset

def load_test_dataset():
    data_files = {"test": "beir_splits/test.jsonl"}
    beir_dataset = load_dataset("json", data_files=data_files)

    dataset_name = "vidore/docvqa_test_subsampled_beir"
    vil_dataset = ViLARMoRDataset(name=dataset_name, load_pseudos=True, load_judgements=True)

    raw_corpus = list(vil_dataset.corpus)  # Make sure it's a list
    corpus_dataset = Dataset.from_list(raw_corpus)

    # Rename 'corpus-id' to 'docid' as expected by collator
    if "corpus-id" in corpus_dataset.column_names:
        corpus_dataset = corpus_dataset.rename_column("corpus-id", "docid")

    return beir_dataset["test"], corpus_dataset, "vidore"

def main():
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    base_model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load adapter weights
    model = PeftModel.from_pretrained(base_model, "./colqwen_beir_checkpoints/checkpoint-21")
    
    from types import MethodType

    # Access the true inner model inside PEFT wrapper
    inner_model = model.base_model.model
    original_inner_forward = inner_model.inner_forward

    def safe_inner_forward(self, *args, **kwargs):
        if "labels" in kwargs:
            kwargs.pop("labels")
        return original_inner_forward(*args, **kwargs)

    # Patch the true method
    inner_model.inner_forward = MethodType(safe_inner_forward, inner_model)


    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False
    )

    # Disable cache and enable gradient checkpointing (if model requires it)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    test_set, corpus_dataset, corpus_format = load_test_dataset()

    collator = CorpusQueryCollator(
        processor=processor,
        max_length=256,
        image_dataset=corpus_dataset,
        mined_negatives=True,
        corpus_format=corpus_format
    )

    training_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=1,
        bf16=True,
        do_train=False,
        do_eval=True,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
    )

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=None,
        eval_dataset=test_set,
        args=training_args,
        data_collator=collator,
        loss_func=ColbertPairwiseNegativeCELoss(),
        is_vision_model=True
    )

    print("Evaluating on test set...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

if __name__ == "__main__":
    main()
