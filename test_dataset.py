from datasets import load_dataset
import os
from vilarmor_dataset import ViLARMoRDataset
from transformers import set_seed

set_seed(42)

name = "vidore/docvqa_test_subsampled_beir"
ds = ViLARMoRDataset(name=name, num_images=2, num_pqueries=5)

def test_query_dataset():
    pq_path = os.path.join(name, "pseudo_queries.json")
    queries = load_dataset("json", data_files=pq_path)
    print(queries)

def test_query_vilarmor_dataset():
    ds_beir = ds.to_beir_dataset()
    print(ds_beir)

def test_dataset_pandas():
    ds_beir = ds.to_beir_dataset()
    queries_ds = ds_beir["queries"]
    queries_df = queries_ds.to_pandas()
    for query_id in queries_ds["query-id"]:
        query = queries_df[queries_df["query-id"] == query_id]["query"].values[0]
        print(f"q({query_id}) = {query}")

def test_get_image():
    corpus_df = ds.corpus.to_pandas()
    break_count = 3
    for corpus_id in corpus_df["corpus-id"]:
        image = ds.get_image(corpus_df, corpus_id)
        print(f"q({corpus_id}) = {image}")
        image.save(f"image_id{corpus_id}.png")
        break_count -= 1
        if(break_count == 0):
            break

def test_get_query():
    queries_df = ds.queries.to_pandas()
    break_count = 3
    for query_id in queries_df["query-id"]:
        query = ds.get_query(queries_df, query_id)
        print(f"q({query_id}) = {query}")
        break_count -= 1
        if(break_count == 0):
            break

if __name__=="__main__":
    # test_query_dataset()
    # test_query_vilarmor_dataset()
    # test_dataset_pandas()
    # test_get_image()
    test_get_query()

