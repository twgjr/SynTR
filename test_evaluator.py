import json
from transformers import set_seed
from vilarmor_dataset import ViLARMoRDataset, COLLECTIONS
from evaluator import ViLARMoREvaluator
from vilarmor_retriever import MODELS, ViLARMoRRetriever

set_seed(42)  # for consistent testing, sets all seeds for randomness

# load the ViLARMoR datasets
model_name = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"
model = ViLARMoRRetriever(model_name)
ds_name = "vidore/docvqa_test_subsampled_beir"
ds = ViLARMoRDataset(name=ds_name, num_images=2, num_pqueries=5)

def test_evaluator_run():
    evaluator = ViLARMoREvaluator(model, ds, 2)
    metrics = evaluator.run()
    print(metrics)

    # # save the evaluation metrics
    # with open("vilarmor_metrics_test.json", "w") as f:
    #     json.dump(metrics, f, indent=4)

if __name__=="__main__":
    test_evaluator_run()