import torch
import numpy as np
from ranx import Run, fuse
from collections import defaultdict

def tensors_to_ranx_runs(score_matrices):
    """
    Convert a list of 2D tensors (one per retriever) into a list of `ranx.Run` objects.

    Args:
        score_matrices (List[torch.Tensor]): List of tensors of shape (num_queries, num_docs)

    Returns:
        List[ranx.Run]: List of `Run` objects for `ranx`
    """
    num_retrievers = len(score_matrices)
    num_queries, num_docs = score_matrices[0].shape

    # Create query and document IDs
    query_ids = [str(i+1) for i in range(num_queries)]
    doc_ids = [str(j+1) for j in range(num_docs)]

    runs = []
    for r in range(num_retrievers):
        score_matrix = score_matrices[r]  # Shape: (num_queries, num_docs)

        # Convert tensor to dictionary format
        retriever_dict = {
            query_ids[q]: {doc_ids[d]: float(score_matrix[q, d]) for d in range(num_docs)}
            for q in range(num_queries)
        }

        # Convert to `ranx.Run`
        runs.append(Run(retriever_dict))

    return runs

def get_global_document_ranking(score_matrices):
    """
    Aggregate document scores across all queries to get a single global ranking.

    Args:
        fused_run (ranx.Run): The fused rankings per query.

    Returns:
        List[Tuple[str, float]]: A sorted list of (document_id, global_score), highest first.
    """

    # Convert tensors to `ranx.Run`
    runs = tensors_to_ranx_runs(score_matrices)

    # Apply RRF
    fused_run = fuse(runs, method="rrf")

    doc_scores = defaultdict(float)

    # Aggregate scores across all queries
    fused_rankings = fused_run.to_dict()
    for query, doc_dict in fused_rankings.items():
        for doc, score in doc_dict.items():
            doc_scores[doc] += score  # Sum scores across queries

    # Sort documents by total aggregated score (highest first)
    sorted_global_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_global_docs