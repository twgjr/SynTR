import qwen
from datasets import Dataset
from dataset import load_local_dataset

SEED = 42

def generate(corpus: Dataset, num_docs=100, num_queries=10):
    """
    Generate pseudo queries and relevance list from sub sample of the corpus.
    """
    model = qwen.load_model()
    processor = qwen.load_processor()
    samples = corpus.shuffle(seed=SEED).select(range(num_docs))
    
    psuedo_queries = [] # ('query-id', 'query')
    pseudo_qrel = [] # ('query-id', 'corpus-id', 'score')

    for d in range(num_docs):
        corpus_id = samples[d]['corpus-id']
        for q in range(num_queries):
            prompt = "Generate a question that the following image can answer. Avoid generating general questions."
            messages = [qwen.message_template(prompt, samples[d]['image'])]
            pseudo_query = qwen.response(model, processor, messages)
            psuedo_queries.append((q, pseudo_query))
            pseudo_qrel.append((q, corpus_id, 1))

    return psuedo_queries, pseudo_qrel


if __name__ == "__main__":
    from dataset import load_local_dataset
    corpus, _, _ = load_local_dataset("vidore/docvqa_test_subsampled_beir")
    psuedo_queries, pseudo_qrel = generate(corpus, num_docs=2, num_queries=2)
    print(psuedo_queries)
    print(pseudo_qrel)