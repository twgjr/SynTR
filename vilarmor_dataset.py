from datasets import load_dataset, Dataset
import os
import json
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


class ViLARMoRDataset:
    def __init__(
        self,
        name: str,
        queries_path:str=None,
        qrels_path:str=None
    ):
        self.name = name
        self._corpus: Dataset = None
        self._queries: Dataset = None
        self.qrels: Dataset = None

        # download the full dataset if it doesn't exist
        if not os.path.exists(self.name):
            corpus_data = self._download_corpus()
            queries_data, qrels_data = self._download_queries_qrels()
            os.makedirs(self.name, exist_ok=False)
            corpus_dir = os.path.join(self.name, "corpus")
            images_dir = os.path.join(corpus_dir, "images")
            os.makedirs(corpus_dir, exist_ok=False)
            os.makedirs(images_dir, exist_ok=False)
            corpus_id_image_list = []

            print(f"Saving images for {self.name}")
            tqdm_images = tqdm(
                corpus_data, desc=f"Saving", total=len(corpus_data)
            )
            for item in tqdm_images:
                item["image-obj"].save(os.path.join(images_dir, f"{item["corpus-id"]}.png"))
                corpus_id_image_list.append(
                    {"corpus-id":item["corpus-id"], 
                     "image-path":item["image-path"]})

            image_map_path = os.path.join(corpus_dir, "corpus_id_image_map.json")
            self._save_data(corpus_id_image_list, image_map_path)
            self._save_data(queries_data, os.path.join(self.name, "queries.json"))
            self._save_data(qrels_data, os.path.join(self.name, "qrels.json"))
            
        if queries_path and qrels_path:
            self.queries, self.qrels = self._load_queries_qrels(
                queries_path=queries_path, qrels_path=qrels_path)

        self.corpus = self._load_corpus_from(None)

    @property
    def corpus(self):
        return self._corpus

    @corpus.setter
    def corpus(self, value):
        self._corpus = value
        self._corpus_index = {int(item["corpus-id"]): i for i, item in enumerate(value)}

    @property
    def queries(self):
        return self._queries

    @queries.setter
    def queries(self, value):
        self._queries = value
        self._query_index = {int(item["query-id"]): i for i, item in enumerate(value)}


    def _download_corpus(self):
        corpus: Dataset = load_dataset(self.name, "corpus")["test"]

        corpus_dir = os.path.join(self.name, "corpus")
        images_dir = os.path.join(corpus_dir, "images")

        corpus_data = []

        print(f"Downloading images for {self.name}")
        tqdm_corpus = tqdm(
            corpus, desc=f"Downloading", total=len(corpus)
        )
        for item in tqdm_corpus:
            image_obj = item["image"]
            image_id = item["corpus-id"]
            image_filename = f"{image_id}.png"
            image_path = os.path.join(images_dir, image_filename)
            corpus_data.append(
                {
                    "corpus-id": image_id,
                    "image-path": image_path,
                    "image-obj": image_obj,
                }
            )

        return corpus_data

    @staticmethod
    def _save_data(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def _download_queries_qrels(self):
        queries: Dataset = load_dataset(self.name, "queries")["test"]
        qrels: Dataset = load_dataset(self.name, "qrels")["test"]

        queries_data = []
        qrels_data = []

        for item in queries:
            queries_data.append({"query-id": item["query-id"], "query": item["query"]})

        for item in qrels:
            qrels_data.append(
                {
                    "query-id": item["query-id"],
                    "corpus-id": item["corpus-id"],
                    "score": item["score"],
                }
            )

        return queries_data, qrels_data

    def _load_corpus_from(self, image_ids: list[int]):
        corpus_dir = os.path.join(self.name, "corpus")
        mapping_path = os.path.join(corpus_dir, "corpus_id_image_map.json")
        corpus_data = []

        with open(mapping_path, "r") as f:
            image_mapping = json.load(f)

        corpus_data = []

        def add_corpus_data(corpus_id, image_path):
            image = Image.open(image_path)
            corpus_item = {"corpus-id":corpus_id, "image":image}
            corpus_data.append(corpus_item)

        if image_ids:
            for corpus_id in image_ids:
                image_path = image_mapping[corpus_id]['image-path']
                add_corpus_data(corpus_id, image_path)
        else:
            for item in image_mapping:
                corpus_id = item['corpus-id']
                image_path = image_mapping[corpus_id]['image-path']
                add_corpus_data(corpus_id, image_path)

        return Dataset.from_list(corpus_data)

    def _load_queries_qrels(self, queries_path: str, qrels_path: str):
        with open(qrels_path, "r") as f:
            qrels = json.load(f)

        with open(queries_path, "r") as f:
            queries = json.load(f)

        return Dataset.from_list(queries), Dataset.from_list(qrels)

    def get_image(self, corpus_id: int):
        idx = self._corpus_index.get(corpus_id)
        if idx is None:
            raise ValueError(f"corpus_id {corpus_id} not found in index.")
        item = self.corpus[idx]
        actual_id = int(item["corpus-id"])
        if actual_id != corpus_id:
            raise ValueError(f"ID mismatch in corpus: expected {corpus_id}, found {actual_id}")
        return item["image"]

    def get_query(self, query_id: int):
        idx = self._query_index.get(query_id)
        if idx is None:
            raise ValueError(f"query_id {query_id} not found in index.")
        item = self.queries[idx]
        actual_id = int(item["query-id"])
        if actual_id != query_id:
            raise ValueError(f"ID mismatch in queries: expected {query_id}, found {actual_id}")
        return item["query"]


    def image_ids(self):
        return list(self.corpus["corpus-id"])
    
    def query_ids(self):
        return list(self.queries["query-id"])

    def filter_from_split(self, split_path):
        print("Filtering dataset with the provided split.")
        # Load the split (val/test) from path
        dataset_split = load_dataset("json", data_files={"split": split_path})["split"]

        filtered_query_ids = set()

        for sample in dataset_split:
            query_id = sample['query-id']
            filtered_query_ids.add(query_id)
                
        # Apply filtering using HuggingFace Dataset filter method
        self.queries = self.queries.filter(
            lambda qry: qry["query-id"] in filtered_query_ids)
        print(f"Filtered queries= {self.queries}")


def merge_and_clean_qrels(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        qrels_1 = json.load(f1)
        qrels_2 = json.load(f2)

    # Merge the lists
    merged_qrels = qrels_1 + qrels_2

    # Track scores by (query-id, corpus-id)
    entry_scores = defaultdict(set)
    for entry in merged_qrels:
        key = (entry['query-id'], entry['corpus-id'])
        entry_scores[key].add(entry['score'])

    # Identify conflicts
    conflict_keys = {key for key, scores in entry_scores.items() if len(scores) > 1}


    # Track seen (query-id, corpus-id, score) triples to avoid duplicates
    seen_entries = set()
    cleaned_qrels = []
    for entry in merged_qrels:
        query_id = entry['query-id']
        key = (query_id, entry['corpus-id'])
        full_key = (query_id, entry['corpus-id'], entry['score'])

        if (key not in conflict_keys) and (full_key not in seen_entries):
            cleaned_qrels.append(entry)
            seen_entries.add(full_key)

    # Determine which query-ids have at least one score == 1
    query_has_positive = {}
    for entry in cleaned_qrels:
        if entry['score'] == 1:
            query_has_positive[entry['query-id']] = True

    cleaned_qrels_no_missing = []
    for entry in cleaned_qrels:
        query_id = entry['query-id']
        key = (query_id, entry['corpus-id'])
        full_key = (query_id, entry['corpus-id'], entry['score'])

        if query_id in query_has_positive:
            cleaned_qrels_no_missing.append(entry)

    # Save cleaned merged data
    with open(output_path, 'w') as outfile:
        json.dump(cleaned_qrels_no_missing, outfile, indent=2)

    removed_entries = len(merged_qrels) - len(cleaned_qrels_no_missing)
    print(f"Removed {removed_entries} entries (conflicts, 0s without 1s, duplicates, queries missing positives).")
    print(f"Cleaned data saved to: {output_path}")


def report_on_qrels(qrels_path):
    qrels = []
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)

    qrel_report_map = {}
    #init entries
    for qrel in qrels:
        qrel_report_map[qrel['query-id']] = {"pos":0, "neg":0}

    # populate
    for qrel in qrels:
        if qrel['score'] == 0:
            qrel_report_map[qrel['query-id']]["neg"] += 1
        elif qrel['score'] > 0:
            qrel_report_map[qrel['query-id']]["pos"] += 1

    # collect summary statisticts
    total_pos = 0
    no_pos = [] # queries with no positives
    no_neg = [] # queries with no negatives
    q_count = 0
    more_docs = []
    max_neg_per_q = None
    min_neg_per_q = None

    for qid in qrel_report_map:
        q_count += 1
        num_pos = qrel_report_map[qid]["pos"]
        num_neg = qrel_report_map[qid]["neg"]

        if max_neg_per_q is None:
            max_neg_per_q = num_neg
        else:
            max_neg_per_q = max(max_neg_per_q, num_neg)

        if min_neg_per_q is None:
            min_neg_per_q = num_neg
        else:
            min_neg_per_q = min(min_neg_per_q, num_neg)

        if num_pos > 0:
            total_pos += num_pos
        if num_pos > 1:
            more_docs.append(qid)
        if num_pos == 0:
            no_pos.append(qid)
        if num_neg == 0:
            no_neg.append(qid)

    print(f"{total_pos} total hard positives")
    print(f"{q_count} queries")
    print(f"{len(more_docs)} queries with more than 1 doc: {more_docs}")
    print(f"{len(no_neg)} queries missing negatives: {no_neg}")
    print(f"{len(no_pos)} queries missing positives: {no_pos}")
    print(f"negatives range from {min_neg_per_q} to {max_neg_per_q}")

if __name__ == "__main__":
    out_path="pseudo_query_sets/general_judge-1-pos/pseudo_qrels_judge_1pos_merged.json"
    merge_and_clean_qrels(
        file1_path="pseudo_query_sets/general_judge-1-pos/pseudo_qrels_judge_1pos.json",
        file2_path="pseudo_query_sets/general_judge-hard-3neg/pseudo_qrels.json",
        output_path=out_path,
    )
    report_on_qrels(out_path)

    

