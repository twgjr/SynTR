import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from datasets import load_dataset

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever

BATCH_SIZE = 4

model_names = [
    "Metric-AI_ColQwen2.5-7b-multilingual-v1.0",
    # "Metric-AI_ColQwen2.5-3b-multilingual-v1.0",
    # "yydxlv_colqwen2.5-7b-v0.1",
    # "tsystems_colqwen2-7b-v1.0",
    # "yydxlv_colqwen2-7b-v1.0",
    # "Metric-AI_colqwen2.5-3b-multilingual",
    # "Alibaba-NLP_gme-Qwen2-VL-7B-Instruct",
    # "Metric-AI_ColQwenStella-2b-multilingual",
    # "tsystems_colqwen2-2b-v1.0",
    # "vidore_colqwen2.5-v0.2",
    # "vidore_colqwen2-v1.0",
    # "vidore_colqwen2.5-v0.1",
    # "Alibaba-NLP_gme-Qwen2-VL-2B-Instruct",
    # "vidore_colqwen2-v0.1",
    # "vidore_colsmolvlm-v0.1",
    # "MrLight_dse-qwen2-2b-mrl-v1",
    # "vidore_colpali-v1.3",
    # "vidore_colSmol-500M",
    # "vidore_ColSmolVLM-Instruct-500M-base",
    # "vidore_ColSmolVLM-Instruct-256M-base",
    # "vidore_colSmol-256M",
    # "vidore_colpali-v1.2",
    # "yydxlv_colphi3.5",
    # "MrLight_dse-phi35-vidore-ft",
    # "vidore_colpali2-3b-pt-448",
    # "marco_mcdse-2b-v1",
]

vidore_names = [
    "vidore/arxivqa_test_subsampled",
    "vidore/docvqa_test_subsampled",
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
    return ColIdefics3Processor.from_pretrained(model_name)


def get_model_instance(model_name):
    return ColIdefics3.from_pretrained(
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

if __name__=="__main__":
    for model_name in model_names:
        processor = get_processor_instance(model_name)
        model = get_model_instance(model_name)
        vision_retriever = get_retriever_instance(model, processor)
        vidore_evaluator = get_vidore_evaluator(vision_retriever)
        for vidore_name in vidore_names:
            metrics = test_vidore_evaluator(vidore_name, "test", BATCH_SIZE, vidore_evaluator)
            print(f"Model: {model_name}, Dataset: {vidore_name}, Metrics: {metrics}")