from vlm import BaseVLM
from datasets import Dataset
from vilarmor_dataset import load_local_dataset, COLLECTIONS
import os
import json
from tqdm import tqdm

SEED = 42


class PseudoQueryGenerator(BaseVLM):
    def __init__(self):
        super().__init__()

    def generate(self, dataset_name: str, corpus: Dataset, num_docs=50, num_queries=5):
        """
        Generate pseudo queries and relevance list from sub sample of the corpus.
        """
        samples = corpus.shuffle(seed=SEED).select(range(num_docs))

        psuedo_queries = []  # ('query-id', 'query')
        pseudo_qrel = []  # ('query-id', 'corpus-id', 'score')
        prompt = "Generate a question that the following image can answer. \
            Avoid generating general questions."

        for d in tqdm(range(num_docs), desc=f"Processing {dataset_name}"):
            corpus_id = samples[d]["corpus-id"]
            for q in tqdm(range(num_queries), desc=f"Generating queries"):
                messages = [self.message_template(prompt, samples[d]["image"])]
                pseudo_query = self.response(messages)
                psuedo_queries.append({"query-id": q, "query": pseudo_query})
                pseudo_qrel.append({"query-id": q, "corpus-id": corpus_id, "score": 1})

        return psuedo_queries, pseudo_qrel
