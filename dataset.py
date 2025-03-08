from datasets import load_dataset, load_from_disk
import os

dataset_names = [
    'vidore/arxivqa_test_subsampled_beir',
    # 'vidore/docvqa_test_subsampled_beir',
    # 'vidore/infovqa_test_subsampled_beir',
    # 'vidore/tabfquad_test_subsampled_beir',
    # 'vidore/tatdqa_test_beir',
    # 'vidore/shiftproject_test_beir',
    # 'vidore/syntheticDocQA_artificial_intelligence_test_beir',
    # 'vidore/syntheticDocQA_energy_test_beir',
    # 'vidore/syntheticDocQA_government_reports_test_beir',
    # 'vidore/syntheticDocQA_healthcare_industry_test_beir',
]

def download_dataset(name):
    try:
        corpus = load_dataset(name, 'corpus', split='test')
        queries = load_dataset(name, 'queries', split='test')
        qrels = load_dataset(name, 'qrels', split='test')
    except Exception as e:
        print(f"Failed to download dataset {name}: {e}")

    # save to prefetch the data to speed up the evaluation
    corpus.save_to_disk(os.path.join(name, 'corpus'))
    queries.save_to_disk(os.path.join(name, 'queries'))
    qrels.save_to_disk(os.path.join(name, 'qrels'))

def load_local_dataset(name):
    corpus = load_from_disk(os.path.join(name, 'corpus'))
    queries = load_from_disk(os.path.join(name, 'queries'))
    qrels = load_from_disk(os.path.join(name, 'qrels'))
    return corpus, queries, qrels

if __name__ == '__main__':
    for name in dataset_names:
        if not os.path.exists(name):
            download_dataset(name)
        corpus, queries, qrels = load_local_dataset(name)
        if corpus and queries and qrels:
            print(corpus)
            print(queries)
            print(qrels)
            print('-----------------------------------')