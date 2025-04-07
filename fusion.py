import torch
from ranx import Run, fuse

def tensor_list_to_ranx_runs(score_matrix_list, query_ids, image_ids):
    runs = []
    for score_matrix in score_matrix_list:
        retriever_dict = {}
        for q in range(score_matrix.shape[0]):
            query = str(query_ids[q])
            if query not in retriever_dict:
                retriever_dict[query] = {}
            for d in range(score_matrix.shape[1]):
                doc = str(image_ids[d])
                retriever_dict[query][doc] = float(score_matrix[q, d])
        runs.append(Run(retriever_dict))
    return runs

def get_fusion_per_query(score_matrix_list, query_ids, image_ids):
    """
    Fuses the scores from many models by query. Reciprical Rank Fusion.
    """
    runs = tensor_list_to_ranx_runs(score_matrix_list, query_ids, image_ids)
    fused_run = fuse(runs, method="rrf")
    return fused_run.to_dict()
