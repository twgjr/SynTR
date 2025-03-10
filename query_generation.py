import qwen
from datasets import Dataset
from dataset import load_local_dataset

SEED = 42

def generate_queries(corpus: Dataset, num_docs=100, num_queries):
    model = qwen.load_model()
    processor = qwen.load_processor()
    samples = corpus.shuffle(seed=SEED).select(range(num_docs))
    
    pq_list = []
    for i in range(len(samples)):
        prompt = "Generate a question that the following image can answer. Avoid generating general questions."
        messages = [qwen.message_template(prompt, samples[i]['image'])]
        pseudo_query = qwen.response(model, processor, messages, p_level=0.99)
        pq_list.append(pseudo_query)

    return pq_list

def generate_queries_all():
    

if __name__ == "__main__":
    from dataset import load_local_dataset
    corpus, _, _ = load_local_dataset("vidore/docvqa_test_subsampled_beir")
    queries = generate_queries(corpus, k=2)
    print(queries)