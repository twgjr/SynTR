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

if __name__=="__main__":
    from vilarmor_dataset import ViLARMoRDataset
    gen_top_p=0.9
    gen_temperature=1.0
    gen_num_pqueries = 3
    gen_corpus_sample_size=400
    ds_name="vidore/docvqa_test_subsampled_beir"

    generator = PseudoQueryGenerator(top_p=gen_top_p, temperature=gen_temperature)
    ds = ViLARMoRDataset(name=ds_name,load_pseudos=False, 
                                load_judgements=False)
    pq_path = os.path.join(ds_name, "pseudo_queries2.json")
    pqrel_path = os.path.join(ds_name, "pseudo_qrels2.json")

    prompt="Imagine a user is researching a topic and is looking for \
    documents that could provide helpful background or support. Given \
    the image, generate a search query that would naturally lead a \
    retrieval system to include this document. The query should reflect \
    a topic of interest, not necessarily a direct fact the image contains. \
    Present the query in the form of natural questions. Do not make any \
    commentary about the search query. Only respond with the query."


    if not os.path.exists(pq_path) or not os.path.exists(pqrel_path):
        psuedo_queries, psuedo_qrels = generator.generate(
                                dataset_name=ds_name,
                                corpus=ds.corpus,
                                corpus_sample_size=gen_corpus_sample_size,
                                num_pqueries=gen_num_pqueries, 
                                # prompt=prompt
                            )

        with open(pq_path, "w") as f:
            json.dump(psuedo_queries, f, indent=4)

        with open(pqrel_path, "w") as f:
            json.dump(psuedo_qrels, f, indent=4)