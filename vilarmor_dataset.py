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
    def __init__(self, name, num_images_test, num_pqueries, num_image_samples,
                    temperature=1.0, top_p=0.9):
        self.name = name
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.qrels: Dataset = None

        if not os.path.exists(name):
            # download, generate and save
            corpus = self.download_corpus()
            generator = PseudoQueryGenerator()
            psuedo_queries, gen_qd_pairs = generator.generate(
                dataset_name=self.name, 
                corpus=corpus, 
                top_p=top_p, 
                temperature=temperature, 
                num_images=num_image_samples, 
                num_queries=num_pqueries,
                )
            self.save_pseudos(psuedo_queries, gen_qd_pairs)
        
        self.load_local_dataset()
        if not (num_images_test == None):
            self.corpus = self.corpus.select(range(num_images_test))

    def to_beir_dataset(self) -> BEIRDataset:
        return BEIRDataset(
            corpus=self.corpus,
            queries=self.queries,
            qrels=self.qrels,
        )

    def download_corpus(self):
        try:
            corpus = load_dataset(self.name, "corpus", split="test")
        except Exception as e:
            print(f"Failed to download dataset {self.name}: {e}")

        # save to prefetch the data to speed up the evaluation
        corpus.save_to_disk(os.path.join(self.name, "corpus"))
        return corpus

    def save_pseudos(self, psuedo_queries, gen_qd_pairs):
        # Save to a JSON file
        with open(os.path.join(self.name, "gen_qd_pairs.json"), "w") as f:
            json.dump(gen_qd_pairs, f, indent=4)

        with open(os.path.join(self.name, "pseudo_queries.json"), "w") as f:
            json.dump(psuedo_queries, f, indent=4)

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




    def get_query_image_ids(self):
        gen_pairs_path = os.path.join(self.name, "gen_qd_pairs.json")
        gen_pairs = load_dataset("json", data_files=gen_pairs_path)['train']
        query_ids = gen_pairs['query-id']
        image_ids = list(set(gen_pairs['corpus-id']))
        return query_ids, image_ids


