from transformers import set_seed
from evaluator import ViLARMoREvaluator
from colpali_engine.models import (
    ColQwen2_5,
    ColQwen2_5_Processor,
)

set_seed(42)  # for consistent testing, sets all seeds for randomness

DATASETS = [
    "vidore/docvqa_test_subsampled_beir",
]

MODELS = {
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0": [ColQwen2_5, ColQwen2_5_Processor],
    "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
    "vidore/colqwen2.5-v0.2": [ColQwen2_5, ColQwen2_5_Processor],
}


# download datasets and generate pseudo queries first
evaluator = ViLARMoREvaluator(
    model_conf = MODELS,
)

evaluator.run(
    ds_name=DATASETS[0],
    judge_top_m=100, 
    gen_top_p=0.9,
    gen_temperature=1.0,
    gen_num_pqueries = 10,
    gen_corpus_sample_size=100
)
    