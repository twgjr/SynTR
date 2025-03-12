"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""

from collections import defaultdict
import torch
import heapq

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import (
    BaseViDoReEvaluator,
)

from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import (
    BEIRDataset,
)

from vilarmor_retriever import ViLARMoRRetriever
from judge import ViLARMoRJudge
from vilarmor_dataset import ViLARMoRDataset


class ViLARMoREvaluator(BaseViDoReEvaluator):
    """
    Applies the ViLARMoR evaluation method to evaluate the performance of a
    vision-and-language retrieval model. Designed to be used with the ViDoRe
    BeIR datasets and v2 leaderboard.
    where each dataset contains 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.
    """

    def __init__(
        self,
        vision_retriever: ViLARMoRRetriever,
        vilarmor_ds: ViLARMoRDataset,
    ):
        super().__init__(vision_retriever=vision_retriever)
        self.ds: BEIRDataset = vilarmor_ds.to_beir_dataset()

        # Dataset column names
        self.corpus_id_column = "corpus-id"
        self.query_id_column = "query-id"
        self.query_column = "query"
        self.passage_column = "text_description"
        if self.vision_retriever.use_visual_embedding:
            self.passage_column = "image"
        self.score_column = "score"

    def _get_retrieval_results(
        self,
        query_ids: list[int],
        passage_ids: list[int],
        scores: torch.Tensor,
        top_k: int = None,  # Optional: Limit to top-k results per query
    ) -> dict[str, dict[str, float]]:
        """
        Get the retrieval results using a max-heap to keep the highest scores at the top.

        Args:
            query_ids (list[int]): The list of query IDs.
            passage_ids (list[int]): The list of passage IDs.
            scores (torch.Tensor): The similarity scores between queries and passages.
            top_k (int, optional): If set, keeps only the top-k highest scores per query.

        Returns:
            dict[str, dict[str, float]]: The retrieval results sorted in descending order.
        """
        results = {}  # Dictionary to store query results
        heap_store = {}  # Dictionary to store heaps per query

        for query_idx, query_id in enumerate(query_ids):
            query_key = str(query_id)

            if query_key not in heap_store:
                heap_store[query_key] = []  # Initialize a heap for the query

            for passage_idx, score in enumerate(scores[query_idx]):
                passage_id = str(passage_ids[passage_idx])
                score_value = float(score.item())

                # Use a min-heap with negative scores to simulate a max-heap
                heapq.heappush(heap_store[query_key], (-score_value, passage_id))

                # If a top_k limit is set, maintain only top_k elements
                if top_k and len(heap_store[query_key]) > top_k:
                    heapq.heappop(heap_store[query_key])  # Remove the lowest score

        # Convert heaps into sorted dictionaries
        for query_key, heap in heap_store.items():
            sorted_scores = sorted(
                heap, reverse=True
            )  # Convert back to descending order
            results[query_key] = {pid: -score for score, pid in sorted_scores}

        return results

    def _rank(
        self,
        batch_query: int,
        batch_passage: int,
        batch_score: int | None = None,
        dataloader_prebatch_query: int | None = None,
        dataloader_prebatch_passage: int | None = None,
    ) -> dict[str, dict[str, float]]:
        # Load datasets
        ds_corpus = self.ds["corpus"]
        ds_queries = self.ds["queries"]

        # Get image data
        passage_ids: list[int] = ds_corpus[self.corpus_id_column]

        # Get query data
        query_ids: list[int] = ds_queries[self.query_id_column]

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        # Get the similarity scores
        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )

        # Get the relevant passages and results
        ranking = self._get_retrieval_results(
            query_ids=query_ids,
            passage_ids=passage_ids,
            scores=scores,
        )

        return ranking

    def _rerank(
        self, ranking: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """
        Rerank the results using the judge model by removing irrelevant results.
        """
        judge = ViLARMoRJudge()
        queries = self.ds["queries"]
        corpus = self.ds["corpus"]
        reranking = ranking.copy()
        for query_id, (corpus_id, rank_val) in ranking.items():
            query = queries[query_id]
            image = corpus[corpus_id]

            if not judge.is_relevant(query, image):
                del ranking[query_id][corpus_id]
        return reranking

    def _evaluate(self, reranking: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Comptutes array of retrieval scores such as nDCG@10 for the given
        retriever model and dataset.  The final score for ViLARMoR on a dataset.

        qrels from dataset are the ground truth relevance scores for the pseudo
        queries generated by VLM.

        reranking is the results of the retrieval model after reranking.
        """
        ds_qrels = self.ds["qrels"]

        # Get query relevance data
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            # Convert qrels to have the format expected by MTEB.
            # NOTE: The IDs are stored as integers in the dataset.
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])
            qrels[query_id][corpus_id] = qrel[self.score_column]

        metrics = self.compute_retrieval_scores(
            qrels=qrels,
            results=reranking,
            ignore_identical_ids=False,
        )

        return metrics

    def run(self) -> dict[str, float]:
        """
        Run the ViLARMoR evaluation for the given retriever model and dataset.
        """
        # Get the retrieval results
        ranking = self._rank(
            batch_query=1,
            batch_passage=1,
            batch_score=1,
        )

        # Rerank the results
        reranking = self._rerank(ranking)

        # Evaluate the reranked results
        metrics = self._evaluate(reranking)

        return metrics
