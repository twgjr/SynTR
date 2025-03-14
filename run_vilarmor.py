import json
from transformes import set_seed
from vilarmor_dataset import ViLARMoRDataset, COLLECTIONS
from evaluator import ViLARMoREvaluator
from vilarmor_retriever import MODELS, ViLARMoRRetriever

set_seed(42)  # for consistent testing, sets all seeds for randomness

# load the ViLARMoR datasets
dataset_list = [ViLARMoRDataset(name) for name in COLLECTIONS]

# run the ViLARMoR evaluator for each dataset and retriever model
metrics = {}
for model_name in MODELS:
    model = ViLARMoRRetriever(model_name)
    for dataset in dataset_list:
        evaluator = ViLARMoREvaluator(model, dataset)
        metrics[model_name] = evaluator.run()

# save the evaluation metrics
with open("vilarmor_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
