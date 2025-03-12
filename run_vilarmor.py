import json
from vilarmor_dataset import ViLARMoRDataset, COLLECTIONS
from evaluator import ViLARMoREvaluator
from vilarmor_retriever import MODELS, ViLARMoRRetriever

# load the ViLARMoR datasets
dataset_list = [ViLARMoRDataset(name) for name in COLLECTIONS]

# run the ViLARMoR evaluator for each dataset and retriever model
metrics = {}
for model_name in MODELS:
    model = ViLARMoRRetriever(model_name)
    for dataset in dataset_list:
        evaluator = ViLARMoREvaluator(dataset, model)
        metrics[model_name] = evaluator.run()

# save the evaluation metrics
with open("vilarmor_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
