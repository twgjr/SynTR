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
    model = PeftModel.from_pretrained(base_model, "./colqwen_beir_checkpoints/checkpoint-54-epoch-6-loss-2e-7")

    from types import MethodType

    ##################################################################
    # Monkey-patch inner_forward to safely remove 'labels' if present
    original_forward = model.base_model.forward

    def safe_forward(self, *args, **kwargs):
        if "labels" in kwargs:
            kwargs.pop("labels")
        return original_forward(*args, **kwargs)

    from types import MethodType
    model.base_model.forward = MethodType(safe_forward, model.base_model)
    # End monkey patch
    #################################################################
    
    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False
    )

    DATASETS = [
        "vidore/docvqa_test_subsampled_beir",
    ]

    MODELS = {
        "colqwen_finetuned": [model, processor], # contains instances
        "Metric-AI/ColQwen2.5-3b-multilingual-v1.0": [ColQwen2_5, ColQwen2_5_Processor],
        "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
        "vidore/colqwen2.5-v0.2": [ColQwen2_5, ColQwen2_5_Processor],
    }


    # download datasets and generate pseudo queries first
    evaluator = ViLARMoREvaluator(
        model_conf = MODELS,
    )

    print("Running Vilarmor Evaluator using existing generated queries and qrels")
    evaluator.run_true(ds_name=DATASETS[0])
    print("Done computing retrieval metrics with ViLARMoREvaluator!")

if __name__ == "__main__":
    main()
