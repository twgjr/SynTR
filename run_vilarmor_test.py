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
    "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
}

# download datasets and generate pseudo queries first
evaluator = ViLARMoREvaluator(
    ds_names = DATASETS,
    model_conf = MODELS,
)

evaluator.run(
    top_k=4, 
    top_p=0.9, 
    temperature=1.0, 
    num_pqueries = 2,
    limit_corpus_size=5
)
    