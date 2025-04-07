import json
import os
import random

# Import your VilarmorDataset
from vilarmor_dataset import ViLARMoRDataset

def generate_beir_samples(
    dataset_name: str,
    negatives_per_query: int = 3,
    seed: int = 42,
) -> list:
    """
    Generate BEIR-style samples from the VilarmorDataset.
    
    Each sample is a dictionary with keys:
      - "query": the query text.
      - "positive_passages": a list with one positive corpus-id.
      - "negative_passages": a list with one or more negative corpus-ids.
    """
    random.seed(seed)

    # Instantiate the dataset; here we load true queries (not pseudo) by setting load_pseudos=False.
    # Adjust load_judgements as needed (here we use False for simplicity).
    dataset = ViLARMoRDataset(name=dataset_name, load_pseudos=True, load_judgements=True)

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
    for qid, query_text in query_map.items():
        if qid not in positive_map:
            continue  # Skip queries with no positive documents

        positives = positive_map[qid]
        positive = positives[0]  # Choose the first positive as the gold passage

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
            "positive_passages": [positive],
            "negative_passages": negatives,
        }
        samples.append(sample)

    return samples

def split_and_save_samples(
    samples: list,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    """
    Shuffle samples and split into train, validation, and test sets,
    then save each as a JSONL file in the given output directory.
    """
    random.seed(seed)
    random.shuffle(samples)
    total = len(samples)
    test_count = int(total * test_size)
    val_count = int(total * val_size)
    train_count = total - test_count - val_count

    train_samples = samples[:train_count]
    val_samples = samples[train_count:train_count + val_count]
    test_samples = samples[train_count + val_count:]

    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(filename, data):
        with open(filename, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")

    write_jsonl(train_file, train_samples)
    write_jsonl(val_file, val_samples)
    write_jsonl(test_file, test_samples)

    print(f"Generated {len(train_samples)} train samples, {len(val_samples)} validation samples, and {len(test_samples)} test samples.")
    print(f"Files saved in directory: {output_dir}")

if __name__ == "__main__":
    # Specify your dataset name as used in VilarmorDataset (e.g., "vidore/docvqa_test_subsampled_beir")
    dataset_name = "vidore/docvqa_test_subsampled_beir"

    # Generate BEIR-style samples using your VilarmorDataset.
    samples = generate_beir_samples(dataset_name=dataset_name, negatives_per_query=3, seed=42)

    # Split the samples into train, validation, and test sets and save them.
    output_directory = "beir_splits"
    split_and_save_samples(samples, output_dir=output_directory, test_size=0.2, val_size=0.1, seed=42)