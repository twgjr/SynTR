import json
from transformers import set_seed
from vilarmor_dataset import ViLARMoRDataset
from evaluator import ViLARMoREvaluator
from vilarmor_retriever import ViLARMoRRetriever
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from judge import ViLARMoRJudge

set_seed(42)  # for consistent testing, sets all seeds for randomness

BATCH_SIZE = 1

# load the ViLARMoR datasets
model_name = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"
model_class = ColQwen2_5
processor_class = ColQwen2_5_Processor
model_conf = {model_name: [model_class, processor_class]}
vr = ViLARMoRRetriever(model_name, model_class, processor_class)
ds_name = "vidore/docvqa_test_subsampled_beir"
num_corpus = 2
ds = ViLARMoRDataset(name=ds_name, num_images=num_corpus, num_pqueries=5)

def test_score_single_model_corpus():
    evaluator = ViLARMoREvaluator(
        ds_names=[ds_name],
        model_conf=model_conf,
        num_corpus=num_corpus)
    evaluator.ds = ds
    evaluator.vision_retriever = vr
    score = evaluator.score_single_model_corpus(
        batch_query = BATCH_SIZE,
        batch_image = BATCH_SIZE,
        batch_score = BATCH_SIZE,
    )
    print(score)    


if __name__=="__main__":
    # test_score_single_model_corpus()
    test_pseudo_relevance_judgement()