import torch
from fusion import get_global_document_ranking

# Example: Suppose we have 3 retrievers, 5 queries, 10 documents
num_retrievers = 3
num_queries = 5
num_docs = 10

# Simulate some scores with random values
score_matrices = [torch.rand((num_queries, num_docs)) for _ in range(num_retrievers)]

# Get overall global document ranking
global_doc_ranking = get_global_document_ranking(score_matrices)

# Print the top 10 most important documents overall
print("Top 10 documents overall:")
for rank, (doc, score) in enumerate(global_doc_ranking[:10], 1):
    print(f"{rank}. {doc}: {score:.4f}")