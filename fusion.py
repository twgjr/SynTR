from ranx import Run, fuse
import numpy as np


def tensor_list_to_ranx_runs(score_matrix_list, query_ids, image_ids):
    runs = []
    for score_matrix in score_matrix_list:
        retriever_dict = {}
        for q_idx in range(score_matrix.shape[0]):
            query = str(query_ids[q_idx])
            # Get document indices sorted by descending score
            ranked_doc_indices = np.argsort(-score_matrix[q_idx])
            # Create an ordered list of doc IDs
            ranked_doc_ids = [str(image_ids[i]) for i in ranked_doc_indices]
            retriever_dict[query] = ranked_doc_ids
        # Ranx will automatically assign descending scores internally
        runs.append(Run(retriever_dict))
    return runs


def get_fusion_per_query(score_matrix_list, query_ids, image_ids):
    """
    Fuses the scores from many models by query. Reciprical Rank Fusion.
    """
    runs = tensor_list_to_ranx_runs(score_matrix_list, query_ids, image_ids)
    fused_run = fuse(runs, method="rrf")

    fused_ordered = {
        query_id: [
            doc_id for doc_id, _ in sorted(
                fused_run[query_id].items(), key=lambda x: -x[1]
            )
        ]
        for query_id in fused_run.get_query_ids()
    }

    return fused_ordered
