import torch
from ranx import Run, fuse

def tensor_list_to_ranx_runs(score_matrix_list, query_ids, image_ids):
    """
    Convert a list of 2D tensors (one per retriever) into a list of `ranx.Run` objects.

    Args:
        score_matrix_list (List[torch.Tensor]): List of tensors (one per retriever).
        query_ids (List[int]): List of query IDs.
        image_ids (List[int]): List of image IDs.

    Returns:
        List[ranx.Run]: List of `Run` objects.
    """
    runs = []
    for score_matrix in score_matrix_list:
        retriever_dict = {
            str(query_ids[q]): {str(image_ids[d]): float(score_matrix[q, d]) for d in range(score_matrix.shape[1])}
            for q in range(score_matrix.shape[0])
        }
        runs.append(Run(retriever_dict))
    return runs

def get_fusion_per_query(score_matrix_list, query_ids, image_ids, save_path=None):
    """
    Perform fusion for each query independently, instead of aggregating scores globally.

    Args:
        score_matrix_list (list of torch.Tensor): List of (queries x documents) score tensors.
        query_ids (list of int): Query identifiers.
        image_ids (list of int): Document identifiers.
        save_path (str, optional): Path to save the fused results.

    Returns:
        dict: Fused ranking per query.
    """
    runs = tensor_list_to_ranx_runs(score_matrix_list, query_ids, image_ids)

    # Perform **query-specific** fusion (not global aggregation)
    fused_run = fuse(runs, method="rrf")

    if save_path:
        fused_run.save(save_path, kind="trec")  # Save in TREC format for compatibility

    return fused_run.to_dict()
