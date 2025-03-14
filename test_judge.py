from vilarmor_dataset import ViLARMoRDataset
from judge import ViLARMoRJudge
from datasets import Dataset, load_dataset
from transformers import set_seed

set_seed(42)

name = "vidore/docvqa_test_subsampled_beir"
ds = ViLARMoRDataset(name=name, num_images=2, num_pqueries=5)
corpus_id_column = 'corpus-id'
passage_column = 'image'
query_id_column = 'query-id'
query_column = 'query'

def test_judge_response():
    judge = ViLARMoRJudge()
    queries_df = ds.queries.to_pandas()
    corpus_df = ds.corpus.to_pandas()

    corpus_id = 473
    image = ds.get_image(corpus_df, corpus_id)

    for query_id in range(10):
        query = ds.get_query(queries_df, query_id)
        judgment = judge.is_relevant(query, image)
        print({"query-id": query_id, "corpus-id": corpus_id, 
            "score": judgment,})
                

if __name__=="__main__":
    test_judge_response()