import os
import torch
from colpali_engine.models import (
    ColQwen2,
    ColQwen2Processor,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColIdefics3,
    ColIdefics3Processor,
    ColPali,
    ColPaliProcessor,
)

from dataset import COLLECTIONS, load_local_dataset
from utils import save_json

from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import (
    ViDoReEvaluatorBEIR,
    BEIRDataset,
)
from vidore_benchmark.retrievers import VisionRetriever

BATCH_SIZE = 4

MODELS = {
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0": [ColQwen2_5, ColQwen2_5_Processor],
    "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
    # below model requires remote code trust, security risk
    # "Metric-AI/ColQwenStella-2b-multilingual":[AutoModel, AutoProcessor],
    "tsystems/colqwen2-2b-v1.0": [ColQwen2, ColQwen2Processor],
    "vidore/colqwen2.5-v0.2": [ColQwen2_5, ColQwen2_5_Processor],
    "vidore/colqwen2-v1.0": [ColQwen2, ColQwen2Processor],
    "vidore/colqwen2.5-v0.1": [ColQwen2_5, ColQwen2_5_Processor],
    "vidore/colqwen2-v0.1": [ColQwen2, ColQwen2Processor],
    "vidore/colsmolvlm-v0.1": [ColIdefics3, ColIdefics3Processor],
    # compatibility error
    # "MrLight/dse-qwen2-2b-mrl-v1": [AutoProcessor, Qwen2VLForConditionalGeneration],
    "vidore/colpali2-3b-pt-448": [ColPali, ColPaliProcessor],
    "vidore/colSmol-500M": [ColIdefics3, ColIdefics3Processor],
    "vidore/ColSmolVLM-Instruct-500M-base": [ColIdefics3, ColIdefics3Processor],
}


def get_processor_instance(model_name):
    processor_class = MODELS[model_name][1]
    try:
        instance = processor_class.from_pretrained(model_name)
    except Exception as e:
        print(f"Problem creating processor: {e}")
    return instance


def get_model_instance(model_name):
    model_class = MODELS[model_name][0]

    try:
        instance = model_class.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        return instance
    except Exception as e:
        print(f"Problem with getting pretrained model: {e}")
        # rethrow
        raise e



def get_retriever_instance(model, processor):
    try:
        instance = VisionRetriever(model=model, processor=processor)
    except Exception as e:
        print(f"Problem with creating Retriever: {e}")
    return instance


def get_vidore_evaluator(vision_retriever):
    try:
        evaluator = ViDoReEvaluatorBEIR(vision_retriever)
    except Exception as e:
        print(f"Problem with creating Evaluator: {e}")
    return evaluator


def test_vidore_evaluator(
    dataset_name, batch_size, vidore_evaluator: ViDoReEvaluatorBEIR, use_pseudo
):
    # Evaluate on a single dataset
    corpus, queries, qrels = load_local_dataset(dataset_name, use_pseudo)
    ds = BEIRDataset(corpus=corpus, queries=queries, qrels=qrels)
    try:
        metrics = vidore_evaluator.evaluate_dataset(
            ds=ds,
            batch_query=batch_size,
            batch_passage=batch_size,
            batch_score=batch_size,
        )
    except Exception as e:
        print(f"Problem running ViDoReEvaluatorBEIR: {e}")
    return metrics


def evaluate_all_models(metrics_dir, use_pseudo):
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    for model_name in MODELS:
        metrics_model_dir = os.path.join(metrics_dir, model_name)
        if not os.path.exists(metrics_model_dir):
            os.makedirs(metrics_model_dir)
            processor = get_processor_instance(model_name)
            model = get_model_instance(model_name)
            vision_retriever = get_retriever_instance(model, processor)
            vidore_evaluator = get_vidore_evaluator(vision_retriever)
            for vidore_path in COLLECTIONS:
                vidore_name = os.path.basename(vidore_path)
                metrics_file_path = os.path.join(metrics_model_dir, vidore_name + ".json")
                if not os.path.exists(metrics_file_path):
                    metrics = test_vidore_evaluator(
                        vidore_path, BATCH_SIZE, vidore_evaluator, use_pseudo
                    )
                    save_json(metrics, metrics_file_path)


if __name__ == "__main__":
    # skips any model/collection pair that is already evaluated
    evaluate_all_models("metrics", use_pseudo=False)
    evaluate_all_models("psuedo_metrics", use_pseudo=True)
