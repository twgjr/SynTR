import qwen
from datasets import Dataset

SEED = 42


def generate_queries(corpus: Dataset, k=100):
    model = qwen.load_model()
    processor = qwen.load_processor()
    samples = corpus.shuffle(seed=SEED).select(range(k))
    
    pq_list = []
    for i in range(len(samples)):
        prompt = "Generate a question that the following image can answer. Avoid generating general questions."
        messages = [qwen.message_template(prompt, samples[i]['image'])]
        pseudo_query = qwen.response(model, processor, messages, p_level=0.9)
        pq_list.append(pseudo_query)

    return pq_list

if __name__ == "__main__":
    from datasets import load_dataset
    corpus = load_dataset("vidore/docvqa_test_subsampled_beir")
    queries = generate_queries(corpus, k=2)
    print(queries)