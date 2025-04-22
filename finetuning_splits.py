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
    Generate BEIR-style samples from the VilarmorDataset.
    
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

    # Build a set of all corpus IDs from the corpus.
    all_corpus_ids = {item["corpus-id"] for item in corpus}

    # Build a mapping from query-id to query text.
    query_map = {q["query-id"]: q["query"] for q in queries}

    # Build a mapping from query-id to a list of positive corpus ids.
    positive_map = {}
    for qrel in qrels:
        qid = qrel["query-id"]
        # Assume a positive judgment is indicated by a score > 0
        if qrel["score"] > 0:
            positive_map.setdefault(qid, []).append(qrel["corpus-id"])

    samples = []
    # For each query that has at least one positive
    for qid, query_text in tqdm(query_map.items(), desc="Building training splits"):
        if qid not in positive_map:
            continue  # Skip queries with no positive documents

        positives = positive_map[qid]
        positive = positives[0]  # Choose the first positive as the gold passage

        if use_hard_neg:
            # Get hard negatives from qrels
            negatives = [
                qrel["corpus-id"]
                for qrel in qrels
                if qrel["query-id"] == qid and qrel["score"] == 0
            ]

            if len(negatives) < negatives_per_query:
                raise ValueError(f"Not enough hard negatives for query-id {qid}: needed {negatives_per_query}, found {len(negatives)}")
        else:
            # randomly choose non-positives as hard negatives
            # Negative candidates: all corpus ids excluding those judged positive for this query.
            negative_candidates = list(all_corpus_ids - set(positives))
            if not negative_candidates:
                # If no negatives available, skip this query.
                continue

            # Sample negatives (if available, up to negatives_per_query)
            num_negatives = min(negatives_per_query, len(negative_candidates))
            negatives = random.sample(negative_candidates, num_negatives)

        sample = {
            "query": query_text,
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
    query_set_dir="pseudo_query_sets"

    set_label="specific_no-judge"
    query_label_dir = os.path.join(query_set_dir, set_label)
    make_splits(use_hard_neg=False, 
                dataset_dir=dataset_dir,
                queries_path=os.path.join(query_label_dir, "pseudo_queries.json"), 
                qrels_path=os.path.join(query_label_dir, "pseudo_qrels.json"), 
                out_dir=os.path.join(splits_dir, set_label))

    set_label="general_no-judge"
    query_label_dir = os.path.join(query_set_dir, set_label)
    make_splits(use_hard_neg=False, 
                dataset_dir=dataset_dir,
                queries_path=os.path.join(query_label_dir, "pseudo_queries.json"), 
                qrels_path=os.path.join(query_label_dir, "pseudo_qrels.json"), 
                out_dir=os.path.join(splits_dir, set_label))

    set_label="general_judge-hard-neg"
    query_label_dir = os.path.join(query_set_dir, set_label)
    make_splits(use_hard_neg=True, # Judge made hard negatives in this set
                dataset_dir=dataset_dir,
                queries_path=os.path.join(query_label_dir, "pseudo_queries.json"), 
                qrels_path=os.path.join(query_label_dir, "pseudo_qrels_merged.json"), 
                out_dir=os.path.join(splits_dir, set_label))