from vilarmor_dataset import ViLARMoRDataset
from pseudo_query import PseudoQueryGenerator
from datasets import Dataset, load_dataset
from transformers import set_seed

set_seed(42)

name = "vidore/docvqa_test_subsampled_beir"
ds = ViLARMoRDataset(name=name, num_images=2, num_pqueries=5)

def test_query_diversity():
    generator = PseudoQueryGenerator()
    psuedo_queries, dq_pairs = generator.generate(
        dataset_name=name, corpus=ds.corpus, num_docs=2, num_queries=5, 
        top_p=0.90, temperature=1.0
    )
    print(psuedo_queries)
    print(dq_pairs)

if __name__=="__main__":
    test_query_diversity()