import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from datasets import load_dataset
from tqdm import tqdm

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever
from vidore_benchmark.utils.data_utils import get_datasets_from_collection

model_name = "vidore/colSmol-256M"
processor = ColIdefics3Processor.from_pretrained(model_name)
model = ColIdefics3.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()

# Get retriever instance
vision_retriever = VisionRetriever(model=model, processor=processor)
vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)

# Evaluate on a single dataset
ds = load_dataset("vidore/tabfquad_test_subsampled", split="test")
metrics_dataset = vidore_evaluator.evaluate_dataset(
    ds=ds,
    batch_query=4,
    batch_passage=4,
)
print(metrics_dataset)

# Evaluate on a local directory or a HuggingFace collection
dataset_names = get_datasets_from_collection("vidore/vidore-benchmark-667173f98e70a1c0fa4db00d")
metrics_collection = {}
for dataset_name in tqdm(dataset_names, desc="Evaluating dataset(s)"):
    metrics_collection[dataset_name] = vidore_evaluator.evaluate_dataset(
        ds=load_dataset(dataset_name, split="test"),
        batch_query=4,
        batch_passage=4,
    )
print(metrics_collection)