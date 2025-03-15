from datasets import load_dataset, load_from_disk
import os
import json
from PIL import Image
import io
from pandas import DataFrame
from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import (
    BEIRDataset,
)
from pseudo_query import PseudoQueryGenerator
from datasets import Dataset

class ViLARMoRDataset:
    def __init__(self, name, image_id_list: list[int] | None):
        self.name = name
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.qrels: Dataset = None
        
        self.load_local_dataset()

        if not (image_id_list == None):
            self.corpus = self.corpus.filter(lambda example: example["corpus-id"] in image_id_list)

    def to_beir_dataset(self) -> BEIRDataset:
        return BEIRDataset(
            corpus=self.corpus,
            queries=self.queries,
            qrels=self.qrels,
        )

    def load_local_dataset(self):
        self.corpus = load_from_disk(os.path.join(self.name, "corpus"))
        pq_path = os.path.join(self.name, "pseudo_queries.json")
        queries = load_dataset("json", data_files=pq_path)
        self.queries = queries['train']
        pqrel_path = os.path.join(self.name, "pseudo_qrel.json")
        if os.path.exists(pqrel_path):
            qrels = load_dataset("json", data_files=pqrel_path)
            self.qrels = qrels['train']

    def get_image(self, corpus_df: DataFrame, corpus_id: int):
        # Ensure 'corpus-id' column is an integer for proper matching
        corpus_id = int(corpus_id)
        corpus_df['corpus-id'] = corpus_df['corpus-id'].astype(int)

        filtered_images = corpus_df[corpus_df['corpus-id'] == corpus_id]

        if filtered_images.empty:
            debug_info = corpus_df.head(3)  # Get top 3 corpus entries for debugging
            raise ValueError(
                f"No image found for corpus_id: {corpus_id}\n"
                f"Top 3 available corpus-ids:\n{debug_info[['corpus-id']].to_string(index=False)}"
            )

        # More robust indexing
        image_map = filtered_images.iloc[0]['image']

        if not isinstance(image_map, dict) or "bytes" not in image_map:
            raise ValueError(
                f"Image data is missing 'bytes' key for corpus_id: {corpus_id}\n"
                f"Available keys in image data: {list(image_map.keys()) if isinstance(image_map, dict) else 'Not a dict'}"
            )

        try:
            image = Image.open(io.BytesIO(image_map["bytes"]))
        except Exception as e:
            raise ValueError(
                f"Failed to open image for corpus_id: {corpus_id}. Error: {e}\n"
                f"First 3 corpus entries: {corpus_df.head(3)[['corpus-id']].to_string(index=False)}"
            )

        return image




    def get_query(self, queries_df: DataFrame, query_id: int):
        query_id = int(query_id)
        queries_df['query-id'] = queries_df['query-id'].astype(int)
        
        filtered_queries = queries_df[queries_df['query-id'] == query_id]

        if filtered_queries.empty:
            debug_info = queries_df.head(3)  # Get top 3 queries for debugging
            raise ValueError(
                f"No query found for query_id: {query_id}\n"
                f"Top 3 available query-ids:\n{debug_info[['query-id', 'query']].to_string(index=False)}"
            )

        return filtered_queries.iloc[0]['query']  # More robust indexing



    @staticmethod
    def get_query_image_ids(name):
        with open(os.path.join(name,"query_ids.json"), "r") as file:
            query_ids = json.load(file)
        with open(os.path.join(name,"image_ids.json"), "r") as file:
            image_ids = json.load(file)
        return query_ids, image_ids


