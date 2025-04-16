from vlm import BaseVLM
from datasets import Dataset
import os
import json
from tqdm import tqdm
from PIL import Image
import random

from transformers import set_seed

set_seed(42)


class PseudoQueryGenerator(BaseVLM):
    def __init__(
        self, top_p: float = 0.9, temperature: float = 1.0):
        super().__init__()
        self.top_p = top_p
        self.temperature = temperature

    def generate(
        self,
        dataset_name: str,
        corpus: Dataset,
        corpus_sample_size:int,
        num_pqueries:int,
        prompt:str="Generate a question that the following image can answer. \
            Avoid generating general questions."
    ):
        """
        Generate pseudo queries and qrel from corpus.
        """
        psuedo_queries = []  # ('query-id', 'query')
        psuedo_qrels = []  # ('query-id', 'corpus-id', 'relevance')

        sampled_images = random.sample(range(len(corpus)), k=corpus_sample_size)

        print(f"Generating queries for {dataset_name}")
        for s, corpus_index in enumerate(tqdm(sampled_images, desc=f"Generating")):
            # randomly choose image from corpus up to sample size times
            corpus_id = corpus[corpus_index]["corpus-id"]
            image = corpus[corpus_index]["image"]

            for q in range(num_pqueries):
                qid = q + s * num_pqueries
                messages = [self.message_template(prompt, image)]
                pseudo_query = self.response(messages, self.top_p, self.temperature)
                psuedo_queries.append({"query-id": qid, "query": pseudo_query})
                psuedo_qrels.append({"query-id": qid, "corpus-id": corpus_id, "score": 1})

        return psuedo_queries, psuedo_qrels
