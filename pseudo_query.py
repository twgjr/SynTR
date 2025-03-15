from vlm import BaseVLM
from datasets import Dataset
import os
import json
from tqdm import tqdm

from transformers import set_seed

set_seed(42)

class PseudoQueryGenerator(BaseVLM):
    def __init__(self):
        super().__init__()

    def generate(self, dataset_name: str, corpus: Dataset, top_p:float, 
                    temperature:float, num_images: int, num_queries: int):
        """
        Generate pseudo queries and relevance list from sub sample of the corpus.
        """
        samples = corpus.shuffle().select(range(num_images))

        psuedo_queries = []  # ('query-id', 'query')
        image_ids = set()
        query_ids = set()
        prompt = "Generate a question that the following image can answer. \
            Avoid generating general questions."

        for d in tqdm(range(num_images), desc=f"Processing {dataset_name}"):
            corpus_id = samples[d]["corpus-id"]
            image_ids.add(corpus_id)
            
            for q in tqdm(range(num_queries), desc=f"Generating queries"):
                qid = q + d*num_queries
                query_ids.add(qid)
                messages = [self.message_template(prompt, samples[d]["image"])]
                pseudo_query = self.response(messages, top_p, temperature)
                psuedo_queries.append({"query-id": qid, "query": pseudo_query})

        image_ids = list(image_ids)
        query_ids = list(query_ids)

        with open(os.path.join(dataset_name, "pseudo_queries.json"), "w") as f:
            json.dump(psuedo_queries, f, indent=4)

        with open(os.path.join(dataset_name, "query_ids.json"), "w") as f:
            json.dump(query_ids, f, indent=4)

        with open(os.path.join(dataset_name, "image_ids.json"), "w") as f:
            json.dump(image_ids, f, indent=4)

        return psuedo_queries, query_ids, image_ids
