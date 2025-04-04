from vlm import BaseVLM
from datasets import Dataset
import os
import json
from tqdm import tqdm

from transformers import set_seed

set_seed(42)


class PseudoQueryGenerator(BaseVLM):
    def __init__(
        self, top_p: float = 0.9, temperature: float = 1.0, num_queries: int = 2
    ):
        super().__init__()
        self.top_p = top_p
        self.temperature = temperature
        self.num_queries = num_queries

    def generate(
        self,
        dataset_name: str,
        corpus: Dataset,
    ):
        """
        Generate pseudo queries and qrel from corpus.
        """
        psuedo_queries = []  # ('query-id', 'query')
        qrels = []  # ('query-id', 'corpus-id', 'relevance')

        prompt = "Generate a question that the following image can answer. \
            Avoid generating general questions."

        for d in tqdm(range(len(corpus)), desc=f"Processing {dataset_name}"):
            corpus_id = corpus[d]["corpus-id"]

            for q in tqdm(range(self.num_queries), desc=f"Generating queries"):
                qid = q + d * self.num_queries
                messages = [self.message_template(prompt, corpus[d]["image"])]
                pseudo_query = self.response(messages, self.top_p, self.temperature)
                psuedo_queries.append({"query-id": qid, "query": pseudo_query})
                qrels.append({"query-id": qid, "corpus-id": corpus_id, "relevance": 1})

        return psuedo_queries, qrels
