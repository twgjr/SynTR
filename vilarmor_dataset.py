from datasets import load_dataset, load_from_disk, Dataset
import os
import json
from pseudo_query import PseudoQueryGenerator
from PIL import Image
import io

class ViLARMoRDataset:
    def __init__(self, name:str, corpus_size_limit:int=None, 
                 generator: PseudoQueryGenerator = None):
        self.name = name
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.qrels: Dataset = None

        if(os.path.exists(os.path.join(self.name, "corpus"))):
            self.load_corpus()

        if self.corpus is None:
            self.download_corpus(name, corpus_size_limit)

        if generator is not None:
            pseudo_queries, pseudo_qrel = generator.generate(
                dataset_name=name,
                corpus=self.corpus,
            )
            self.queries = pseudo_queries
            self.qrels = pseudo_qrel

            with open(os.path.join(name, "pseudo_queries.json"), "w") as f:
                json.dump(pseudo_queries, f, indent=4)

            with open(os.path.join(name, "pseudo_pqrels.json"), "w") as f:
                json.dump(pseudo_qrel, f, indent=4)
        else:
            self.load_queries()
            self.load_qrels()

        if self.queries is None or self.qrels is None:
            raise ValueError(
                "Queries or qrels not found.\n"
                "Provide a generator to generate pseudo queries and qrels.\n")
            
    def download_corpus(self, name, corpus_size_limit):
        try:
            corpus: Dataset = load_dataset(name, "corpus", split="test")
        except Exception as e:
            print(f"Failed to download dataset {name}: {e}")

        if corpus_size_limit:
            # Limit the size of the corpus
            corpus = corpus.select(range(corpus_size_limit))

        # save to prefetch the data to speed up the evaluation
        corpus.save_to_disk(os.path.join(name, "corpus"))
        
        self.corpus = corpus

    def load_corpus(self):
        self.corpus = load_from_disk(os.path.join(self.name, "corpus"))

    def load_queries(self):
        pq_path = os.path.join(self.name, "pseudo_queries.json")
        queries = load_dataset("json", data_files=pq_path)
        self.queries = queries["train"]

    def load_qrels(self):
        pqrel_path = os.path.join(self.name, "pseudo_pqrels.json")
        qrels = load_dataset("json", data_files=pqrel_path)
        self.qrels = qrels["train"]

    def get_image(self, corpus_id: int):
        # Filter the dataset to find items with matching corpus-id
        filtered_images = []
        for item in self.corpus:
            if int(item["corpus-id"]) == corpus_id:
                filtered_images.append(item)

        if not filtered_images:
            raise ValueError(f"No image found for corpus_id: {corpus_id}\n")

        if len(filtered_images) > 1:
            raise ValueError(f"Duplicate corpus_id found: {corpus_id}\n")

        # Extract the image map from the only matching item
        image_map = filtered_images[0]["image"]

        try:
            image = Image.open(io.BytesIO(image_map["bytes"]))
        except Exception as e:
            raise ValueError(
                f"Failed to open image for corpus_id: {corpus_id}. Error: {e}\n"
            )

        return image

    def get_query(self, query_id: int):
        # Filter the dataset to find items with matching query-id
        filtered_queries = []
        for item in self.queries:
            if int(item["query-id"]) == query_id:
                filtered_queries.append(item)

        if not filtered_queries:
            raise ValueError(f"No query found for query_id: {query_id}\n")

        if len(filtered_queries) > 1:
            raise ValueError(f"Duplicate query_id found: {query_id}\n")

        return filtered_queries[0]["query"]

    def get_query_image_ids(self):
        query_ids = set()
        image_ids = set()
        for qrel in self.qrels:
            query_ids.add(qrel["query-id"])
            image_ids.add(qrel["corpus-id"])
        return list(query_ids), list(image_ids)
