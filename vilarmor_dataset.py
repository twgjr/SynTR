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
    """
    Merges two QREL-style JSON files and removes entries with conflicting scores.
    
    Parameters:
    - file1_path: str, path to the first JSON file
    - file2_path: str, path to the second JSON file
    - output_path: str, path where the cleaned merged output should be saved
    """
    # Load both JSON files
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

    # Filter out conflicting entries
    cleaned_qrels = [
        entry for entry in merged_qrels
        if (entry['query-id'], entry['corpus-id']) not in conflict_keys
    ]

    # Save cleaned merged data
    with open(output_path, 'w') as outfile:
        json.dump(cleaned_qrels, outfile, indent=2)

    print(f"Removed {len(merged_qrels) - len(cleaned_qrels)} conflicting entries.")
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    # base_dir='general-pseudo-queries_judge'
    # file1='pseudo_qrels_judge_negatives.json'
    # file2='pseudo_qrels_positives.json'
    # output_file='cleaned_merged_qrels.json'

    # merge_and_clean_qrels(
    #     file1_path=os.path.join(base_dir, file1), 
    #     file2_path=os.path.join(base_dir, file2), 
    #     output_path=os.path.join(base_dir, output_file)
    # )
    pass
