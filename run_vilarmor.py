import json
from transformers import set_seed
from evaluator import ViLARMoREvaluator
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

set_seed(42)  # for consistent testing, sets all seeds for randomness

DATASETS = [
    "vidore/docvqa_test_subsampled_beir",
    # "vidore/tatdqa_test_beir",
]

MODELS = {
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0": [ColQwen2_5, ColQwen2_5_Processor],
    # "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
    ## below model requires remote code trust, security risk
    ## "Metric-AI/ColQwenStella-2b-multilingual":[AutoModel, AutoProcessor],
    # "tsystems/colqwen2-2b-v1.0": [ColQwen2, ColQwen2Processor],
    # "vidore/colqwen2.5-v0.2": [ColQwen2_5, ColQwen2_5_Processor],
    # "vidore/colqwen2-v1.0": [ColQwen2, ColQwen2Processor],
    # "vidore/colqwen2.5-v0.1": [ColQwen2_5, ColQwen2_5_Processor],
    # "vidore/colqwen2-v0.1": [ColQwen2, ColQwen2Processor],
    # "vido/re/colsmolvlm-v0.1": [ColIdefics3, ColIdefics3Processor],
    ## compatibility error
    ## "MrLight/dse-qwen2-2b-mrl-v1": [AutoProcessor, Qwen2VLForConditionalGeneration],
    # "vidore/colpali2-3b-pt-448": [ColPali, ColPaliProcessor],
    # "vidore/colSmol-500M": [ColIdefics3, ColIdefics3Processor],
    # "vidore/ColSmolVLM-Instruct-500M-base": [ColIdefics3, ColIdefics3Processor],
}


evaluator = ViLARMoREvaluator(ds_names=DATASETS, model_names=MODELS)