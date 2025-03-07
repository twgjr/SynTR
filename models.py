import os
import json
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

from datasets import load_dataset

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever

BATCH_SIZE = 4

models = {
    ## "Metric-AI/ColQwen2.5-7b-multilingual-v1.0":[ColQwen2_5,ColQwen2_5_Processor],
    ## "Metric-AI/ColQwen2.5-3b-multilingual-v1.0":[ColQwen2_5,ColQwen2_5_Processor],
    ## "yydxlv/colqwen2.5-7b-v0.1":[ColQwen2_5,ColQwen2_5_Processor],
    "tsystems/colqwen2-7b-v1.0": [ColQwen2, ColQwen2Processor],
    # "yydxlv/colqwen2-7b-v1.0": [ColQwen2, ColQwen2Processor],
    ## "Metric-AI/colqwen2.5-3b-multilingual":[ColQwen2_5,ColQwen2_5_Processor],
    # "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    # "Metric-AI/ColQwenStella-2b-multilingual",
    # "tsystems/colqwen2-2b-v1.0": [ColQwen2, ColQwen2Processor],
    ## "vidore/colqwen2.5-v0.2":[ColQwen2_5,ColQwen2_5_Processor],
    # "vidore/colqwen2-v1.0": [ColQwen2, ColQwen2Processor],
    ## "vidore/colqwen2.5-v0.1":[ColQwen2_5,ColQwen2_5_Processor],
    # "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    # "vidore/colqwen2-v0.1": [ColQwen2, ColQwen2Processor],
    # "vidore/colsmolvlm-v0.1",
    # "MrLight/dse-qwen2-2b-mrl-v1",
    # "vidore/colpali-v1.3",
    # "vidore/colSmol-500M",
    # "vidore/ColSmolVLM-Instruct-500M-base",
    # "vidore/ColSmolVLM-Instruct-256M-base",
    # "vidore/colSmol-256M",
    # "vidore/colpali-v1.2",
    # "yydxlv/colphi3.5",
    # "MrLight/dse-phi35-vidore-ft",
    # "vidore/colpali2-3b-pt-448",
    # "marco/mcdse-2b-v1",
}

vidore_names = [
    "vidore/arxivqa_test_subsampled",
    # "vidore/docvqa_test_subsampled",
    # "vidore/infovqa_test_subsampled",
    # "vidore/tatdqa_test",
    # "vidore/tabfquad_test_subsampled",
    # "vidore/syntheticDocQA_artificial_intelligence_test",
    # "vidore/syntheticDocQA_government_reports_test",
    # "vidore/syntheticDocQA_healthcare_industry_test",
    # "vidore/syntheticDocQA_energy_test",
    # "vidore/shiftproject_test",
]


def get_processor_instance(model_name):
    processor_class = models[model_name][1]
    return processor_class.from_pretrained(model_name)


def get_model_instance(model_name):
    model_class = models[model_name][0]
    return model_class.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()


def get_retriever_instance(model, processor):
    return VisionRetriever(model=model, processor=processor)


def get_vidore_evaluator(vision_retriever):
    return ViDoReEvaluatorQA(vision_retriever)


def test_vidore_evaluator(dataset_name, split, batch_size, vidore_evaluator):
    # Evaluate on a single dataset
    ds = load_dataset(dataset_name, split=split)
    metrics = vidore_evaluator.evaluate_dataset(
        ds=ds, batch_query=batch_size, batch_passage=batch_size, batch_score=batch_size
    )
    return metrics


def save_metrics(metrics, dir):
    path = os.path.join(dir, ".json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate_all_models():
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    for model_name in models:
        metrics_model_dir = os.path.join(metrics_dir, model_name)
        if os.path.exists(metrics_model_dir):
            continue
        os.makedirs(metrics_model_dir)
        processor = get_processor_instance(model_name)
        model = get_model_instance(model_name)
        vision_retriever = get_retriever_instance(model, processor)
        vidore_evaluator = get_vidore_evaluator(vision_retriever)
        for vidore_name in vidore_names:
            metrics_model_vidore_dir = os.path.join(metrics_model_dir, vidore_name)
            if os.path.exists(metrics_model_vidore_dir):
                continue
            os.makedirs(metrics_model_vidore_dir)
            metrics = test_vidore_evaluator(
                vidore_name, "test", BATCH_SIZE, vidore_evaluator
            )
            save_metrics(metrics, metrics_model_vidore_dir)


if __name__ == "__main__":
    evaluate_all_models()
