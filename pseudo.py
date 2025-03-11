import qwen
from datasets import Dataset
from dataset import load_local_dataset, dataset_names
import os
import json
from tqdm import tqdm

SEED = 42


def generate(corpus: Dataset, model, processor, num_docs=50, num_queries=5):
    """
    Generate pseudo queries and relevance list from sub sample of the corpus.
    """
    samples = corpus.shuffle(seed=SEED).select(range(num_docs))

    psuedo_queries = []  # ('query-id', 'query')
    pseudo_qrel = []  # ('query-id', 'corpus-id', 'score')

    for d in tqdm(range(num_docs), desc="Processing image"):
        corpus_id = samples[d]["corpus-id"]
        for q in tqdm(range(num_queries), desc=f"Generating queries"):
            prompt = "Generate a question that the following image can answer. Avoid generating general questions."
            messages = [qwen.message_template(prompt, samples[d]["image"])]
            pseudo_query = qwen.response(model, processor, messages)
            psuedo_queries.append({"query-id": q, "query": pseudo_query})
            pseudo_qrel.append({"query-id": q, "corpus-id": corpus_id, "score": 1})

    return psuedo_queries, pseudo_qrel


def save_pseudos(psuedo_queries, pseudo_qrel, path):
    # Save to a JSON file
    with open(os.path.join(path, "pseudo_qrel_truth.json"), "w") as f:
        json.dump(pseudo_qrel, f, indent=4)

    with open(os.path.join(path, "pseudo_queries.json"), "w") as f:
        json.dump(psuedo_queries, f, indent=4)

def rename_json_field(path, field_name, new_name):
    with open(path, "r") as f:
        data = json.load(f)
    for item in data:
        item[new_name] = item.pop(field_name)
    with open(os.path.join(os.path.dirname(path), os.path.basename(path)+"_new.json"), "w") as f:
        json.dump(data, f, indent=4)


def generate_all():
    NUM_DOCS = 50
    NUM_QUERIES = 5

    model = qwen.load_model()
    processor = qwen.load_processor()

    for name in tqdm(dataset_names, desc="Processing dataset"):
        if os.path.exists(os.path.join(name, "pseudo_qrel_truth.json")) and \
            os.path.exists(os.path.join(name, "pseudo_queries.json")):
            print(f"Pseudo queries and relevance list already exists for {name}. Skipping...")
            continue
        corpus, _, _ = load_local_dataset(name, use_pseudo=False)
        psuedo_queries, pseudo_qrel = generate(
            corpus, model, processor, num_docs=NUM_DOCS, num_queries=NUM_QUERIES
        )
        save_pseudos(psuedo_queries, pseudo_qrel, name)


if __name__ == "__main__":
    # generate_all()
    rename_json_field("vidore/docvqa_test_subsampled_beir/pseudo_qrel_truth.json", "query", "corpus-id")
    rename_json_field("vidore/tadqa_test_beir/pseudo_qrel_truth.json", "query", "corpus-id")
