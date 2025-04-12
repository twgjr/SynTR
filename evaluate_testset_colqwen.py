import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, BitsAndBytesConfig
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.collators import CorpusQueryCollator
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseNegativeCELoss
from peft import PeftModel
from vilarmor_dataset import ViLARMoRDataset
from evaluator import ViLARMoREvaluator

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
    
    # from types import MethodType

    # # Access the true inner model inside PEFT wrapper
    # inner_model = model.base_model.model
    # original_inner_forward = inner_model.inner_forward

    # def safe_inner_forward(self, *args, **kwargs):
    #     if "labels" in kwargs:
    #         kwargs.pop("labels")
    #     return original_inner_forward(*args, **kwargs)

    # # Patch the true method
    # inner_model.inner_forward = MethodType(safe_inner_forward, inner_model)

    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False
    )

    test_set, corpus_dataset, corpus_format = load_test_dataset()


    DATASETS = [
        "vidore/docvqa_test_subsampled_beir",
    ]

    MODELS = {
        "Metric-AI/ColQwen2.5-3b-multilingual-v1.0": [ColQwen2_5, ColQwen2_5_Processor],
        "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
        "vidore/colqwen2.5-v0.2": [ColQwen2_5, ColQwen2_5_Processor],
        "colqwen_finetuned": [model, processor] # contains instances
    }


    # download datasets and generate pseudo queries first
    evaluator = ViLARMoREvaluator(
        model_conf = MODELS,
    )

    print("Running Vilarmor Evaluator using existing generated queries and qrels")
    evaluator.run(
        ds_name=DATASETS[0],
        judge_top_m=5,
        gen_top_p=0.9,
        gen_temperature=1.0,
        gen_num_pqueries = 2,
        gen_corpus_sample_size=50
    )
    print("Done computing retrieval metrics with ViLARMoREvaluator!")

if __name__ == "__main__":
    main()
