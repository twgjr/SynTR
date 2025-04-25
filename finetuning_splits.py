import json
import os
import random
from tqdm import tqdm

# Import your VilarmorDataset
from vilarmor_dataset import ViLARMoRDataset

def generate_beir_samples(
    dataset_dir:str, 
    queries_path:str,
    qrels_path:str,
    negatives_per_query: int = 3,
    seed: int = 42,
    use_hard_neg:bool=True
) -> list:
    """
    Generate BEIR-style samples from the VilarmorDataset. Assumes that every 
    query has at least one positive and negatives_per_query negatives, and 
    no duplicates or conflicts
    
    Each sample is a dictionary with keys:
      - "query": the query text.
      - "positive_passages": a list with one positive corpus-id.
      - "negative_passages": a list with one or more negative corpus-ids.
    """
    random.seed(seed)

    # for loading ground truth dataset
    dataset = ViLARMoRDataset(name=dataset_dir, queries_path=queries_path, qrels_path=qrels_path)

    # Load queries and qrels (both are lists of dictionaries)
    queries = dataset.queries      # each with keys "query-id" and "query"
    qrels = dataset.qrels          # each with keys "query-id", "corpus-id", "score"
    corpus = dataset.corpus        # a Hugging Face Dataset of corpus items (each has "corpus-id" and image data)

    query_map = {}
    for item in queries:
        query_id = item['query-id']
        query = item['query']
        query_map[query_id]=query

    # Init map of queries to positives and negatives
    positive_map = {}
    for qrel in qrels:
        positive_map[qrel["query-id"]] = {"positives":[], "negatives":[]}

    # repeat and add positives and negatives
    for qrel in qrels:
        qid = qrel["query-id"]
        doc_id = qrel["corpus-id"]
        rel_val = qrel['score']
        if rel_val > 0:
            positive_map[qid]['positives'].append(doc_id)
        else:
            positive_map[qid]['negatives'].append(doc_id)

    samples = []
    for qid in positive_map:
        for positive in positive_map[qid]["positives"]:
            negative_candidates = positive_map[qid]["negatives"]
            # randomly sample without replacement from the negatives
            sample_count = min(len(negative_candidates), negatives_per_query)
            negatives = random.sample(negative_candidates, sample_count)

            sample = {
            "query": query_map[qid],
            "query-id": qid,
            "positive_passages": [positive],
            "negative_passages": negatives,
            }
            samples.append(sample)

    return samples

def split_and_save_samples(
    samples: list,
    output_dir: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)
    random.shuffle(samples)
    total = len(samples)

    test_count = int(total * test_size)
    val_count = int(total * val_size)
    train_count = total - val_count - test_count

    train_samples = samples[:train_count]
    val_samples = samples[train_count:train_count + val_count]
    test_samples = samples[train_count + val_count:]

    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(filename, data):
        with open(filename, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    write_jsonl(os.path.join(output_dir, "train.jsonl"), train_samples)
    write_jsonl(os.path.join(output_dir, "val.jsonl"), val_samples)
    write_jsonl(os.path.join(output_dir, "test.jsonl"), test_samples)

    print(f"Generated {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples.")
    print(f"Files saved in directory: {output_dir}")

def make_splits(use_hard_neg, dataset_dir, out_dir, queries_path, qrels_path):
    # Generate BEIR-style samples using your VilarmorDataset.
    samples = generate_beir_samples(
        dataset_dir=dataset_dir, queries_path=queries_path, qrels_path=qrels_path, 
        negatives_per_query=3, seed=42, use_hard_neg=use_hard_neg)

    # Split the samples into train, validation sets and save them.
    split_and_save_samples(samples, output_dir=out_dir, val_size=0.1, seed=42)

if __name__ == "__main__":
    dataset_dir="vidore/docvqa_test_subsampled_beir"
    splits_dir="splits"
    os.makedirs(splits_dir, exist_ok=True)

    make_splits(use_hard_neg=True, # Judge made hard negatives in this set
                dataset_dir=dataset_dir,
                queries_path="pseudo_query_sets/general_judge-1-pos/pseudo_queries.json", 
                qrels_path="pseudo_query_sets/general_judge-1-pos/pseudo_qrels_judge_1pos_merged.json", 
                out_dir="splits/general_judge-1-pos")