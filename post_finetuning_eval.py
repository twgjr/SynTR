import os
import shutil
import json
from tqdm import tqdm
from evaluator import compute_metrics
from fine_tune_colqwen import get_model_and_processor
from vilarmor_dataset import ViLARMoRDataset

def compute_validation(base_dir:str, dataset:ViLARMoRDataset) -> dict:
    from evaluator import compute_metrics
    val_metrics = {}

    checkpoint_dirs = os.listdir(base_dir)
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")

    val_metrics["best_checkpoint"] = None
    val_metrics["best_metric"] = None

    # compute metrics for all checkpoints
    for cp_dir in tqdm(checkpoint_dirs, desc=f"Computing best checkpoint for {base_dir}"):
        full_cp_dir = os.path.join(base_dir, cp_dir)
        model, processor, peft_config = get_model_and_processor(
            checkpoint_dir=full_cp_dir, use_peft=True)
        metrics = compute_metrics(
            model=model, processor=processor,
            dataset=dataset
        )

        metrics = next(iter(metrics.values()))

        if val_metrics["best_metric"] is None:
            val_metrics["best_metric"] = metrics
            val_metrics["best_checkpoint"] = cp_dir
        elif metrics["ndcg_at_10"] > val_metrics["best_metric"]["ndcg_at_10"]:
            val_metrics["best_metric"] = metrics
            val_metrics["best_checkpoint"] = cp_dir

        val_metrics[cp_dir] = metrics

    return val_metrics

def compute_test(base_dir:str, dataset:ViLARMoRDataset, val_metrics:dict=None) -> dict:
    from evaluator import compute_metrics
    test_metrics = {}

    if val_metrics:
        # finetuned model, best checkpoint
        cp_dir = val_metrics["best_checkpoint"]
        full_cp_dir = os.path.join(base_dir, cp_dir)
        model, processor, peft_config = get_model_and_processor(
            checkpoint_dir=full_cp_dir, use_peft=True)
    else:
        # base model
        model, processor, peft_config = get_model_and_processor(
            checkpoint_dir=None, use_peft=False)
        cp_dir="base"

    metrics = compute_metrics(
        model=model, processor=processor,
        dataset=dataset
    )

    metrics = next(iter(metrics.values()))

    test_metrics[cp_dir] = metrics

    return test_metrics

def load_dataset(ds_path, splits_path, pseudo_queries_path, pqrels_path):
    dataset = ViLARMoRDataset(
        name=ds_path, 
        queries_path=os.path.join(pseudo_queries_path), 
        qrels_path=os.path.join(pqrels_path))
    
    dataset.filter_from_split(splits_path)

    return dataset

def eval_set(ds_path, query_set_label, split_set_label, result_set_label):
    val_splits_path = f"splits/{split_set_label}/val.jsonl"
    test_splits_path = f"splits/{split_set_label}/test.jsonl"
    pseudo_queries_path = f"pseudo_query_sets/{query_set_label}/pseudo_queries.json"
    pqrels_path = f"pseudo_query_sets/{query_set_label}/pseudo_qrels.json"
    cp_dir = f"results/{result_set_label}/checkpoints"

    ### Validation ###
    dataset = load_dataset(
        ds_path=ds_path, splits_path=val_splits_path, 
        pseudo_queries_path=pseudo_queries_path, pqrels_path=pqrels_path
    )

    # validation
    val_metrics = compute_validation(base_dir=cp_dir, dataset=dataset)

    # save validation metrics
    with open(f"results/{result_set_label}/val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=4)

    #### Test ###
    dataset = load_dataset(
        ds_path=ds_path, splits_path=test_splits_path, 
        pseudo_queries_path=pseudo_queries_path, pqrels_path=pqrels_path
    )

    # # test fine-tuned model
    test_metrics = compute_test(base_dir=cp_dir, dataset=dataset, val_metrics=val_metrics)
    # save test metrics
    with open(f"results/{result_set_label}/test_metrics_finetuned.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # test base model
    test_metrics = compute_test(base_dir=cp_dir, dataset=dataset, val_metrics=None)
    # save test metrics
    with open(f"results/{result_set_label}/test_metrics_base.json", "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__=="__main__":
    eval_set(ds_path="vidore/docvqa_test_subsampled_beir", 
        split_set_label="general_judge-1-pos",
        query_set_label="general_judge-1-pos",
        result_set_label="general_judge-1-pos")
