import os
import json
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor, \
                                ColQwen2_5, ColQwen2_5_Processor, \
                                ColIdefics3, ColIdefics3Processor, \
                                ColPali, ColPaliProcessor
from transformers import AutoModel, AutoProcessor, \
                        Qwen2VLForConditionalGeneration, \
                        AutoProcessor, AutoModelForCausalLM

from dataset import dataset_names, load_local_dataset

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever

BATCH_SIZE = 4

models = {
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0":[ColQwen2_5,ColQwen2_5_Processor],
    # "Metric-AI/colqwen2.5-3b-multilingual":[ColQwen2_5,ColQwen2_5_Processor],
    # "Metric-AI/ColQwenStella-2b-multilingual":[AutoModel, AutoProcessor],
    # "tsystems/colqwen2-2b-v1.0": [ColQwen2, ColQwen2Processor],
    # "vidore/colqwen2.5-v0.2":[ColQwen2_5,ColQwen2_5_Processor],
    # "vidore/colqwen2-v1.0": [ColQwen2, ColQwen2Processor],
    # "vidore/colqwen2.5-v0.1":[ColQwen2_5,ColQwen2_5_Processor],
    # "vidore/colqwen2-v0.1": [ColQwen2, ColQwen2Processor],
    # "vidore/colsmolvlm-v0.1":[ColIdefics3, ColIdefics3Processor],
    # "MrLight/dse-qwen2-2b-mrl-v1": [AutoProcessor, Qwen2VLForConditionalGeneration],
    # "vidore/colpali-v1.3": [ColPali, ColPaliProcessor],
    # "vidore/colSmol-500M": [ColIdefics3, ColIdefics3Processor],
    # "vidore/ColSmolVLM-Instruct-500M-base": [ColIdefics3, ColIdefics3Processor],
    # "vidore/ColSmolVLM-Instruct-256M-base": [ColIdefics3, ColIdefics3Processor],
    # "vidore/colSmol-256M": [ColIdefics3, ColIdefics3Processor],
    # "vidore/colpali-v1.2": [ColPali, ColPaliProcessor],
    # "yydxlv/colphi3.5": [ColIdefics3, ColIdefics3Processor],
    # "MrLight/dse-phi35-vidore-ft": [AutoProcessor, AutoModelForCausalLM],
    # "vidore/colpali2-3b-pt-448": [ColPali, ColPaliProcessor],
    # "marco/mcdse-2b-v1":[AutoProcessor, Qwen2VLForConditionalGeneration],
}


def get_processor_instance(model_name):
    processor_class = models[model_name][1]
    return processor_class.from_pretrained(model_name)


def get_model_instance(model_name, use_4bit=False, use_cuda=True):
    # all models inherit from PreTrainedModel so support load_in_4bit and 
    # load_in_8bit methods
    model_class = models[model_name][0]
    return model_class.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if use_cuda else "cpu",
        load_in_8bit=use_4bit,
    ).eval()


def get_retriever_instance(model, processor):
    return VisionRetriever(model=model, processor=processor)


def get_vidore_evaluator(vision_retriever):
    return ViDoReEvaluatorQA(vision_retriever)


def test_vidore_evaluator(dataset_name, batch_size, vidore_evaluator: ViDoReEvaluatorQA):
    # Evaluate on a single dataset
    ds = load_local_dataset(dataset_name)
    metrics = vidore_evaluator.evaluate_dataset(
        ds=ds, batch_query=batch_size, batch_passage=batch_size, batch_score=batch_size
    )
    return metrics


def save_metrics(metrics, dir, name):
    path = os.path.join(dir, name, ".json")
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
        model = get_model_instance(model_name, use_4bit=True, use_cuda=False)
        vision_retriever = get_retriever_instance(model, processor)
        vidore_evaluator = get_vidore_evaluator(vision_retriever)
        for vidore_name in dataset_names:
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
