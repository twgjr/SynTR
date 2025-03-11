from vidore import MODELS, get_model_instance, get_processor_instance, get_retriever_instance, get_vidore_evaluator, test_vidore_evaluator
from dataset import load_local_dataset, COLLECTIONS
import faiss
from vidore_benchmark.retrievers import VisionRetriever

BATCH_SIZE = 4

def create_corpus_index(model_name, dataset_name):
    model = get_model_instance(model_name)
    processor = get_processor_instance(model_name)
    retriever:VisionRetriever = get_retriever_instance(model, processor)
    corpus, _, _ = load_local_dataset(dataset_name, use_pseudo=False)
    
    index = faiss.IndexFlatIP(retriever.forward_passages([corpus[0]], batch_size=1).shape[1])
    
    for i in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[i:i + BATCH_SIZE]
        corpus_embeddings = retriever.forward_passages(batch, batch_size=BATCH_SIZE)
        index.add(corpus_embeddings)
    
    return index


if __name__ == "__main__":
    model_name = next(iter(MODELS))
    dataset_name = COLLECTIONS[0]
    index = create_corpus_index(model_name, dataset_name)
    print(index.ntotal)