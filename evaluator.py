"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""
import os
import json
from collections import defaultdict
import torch
import torch.nn.functional as F


from datasets import Dataset
from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import (
    BaseViDoReEvaluator,
)

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


from vilarmor_retriever import ViLARMoRRetriever
from judge import ViLARMoRJudge
from vilarmor_dataset import ViLARMoRDataset
from fusion import get_global_document_ranking

BATCH_SIZE = 1

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
        num_corpus:int=None # subset of images for testing
    ):
        super().__init__(vision_retriever=vision_retriever)
        if(num_corpus):
            vilarmor_ds.corpus = vilarmor_ds.corpus.select(range(num_corpus))
        self.ds: ViLARMoRDataset = vilarmor_ds

        # Dataset column names
        self.corpus_id_column = "corpus-id"
        self.query_id_column = "query-id"
        self.query_column = "query"
        self.passage_column = "image"
        self.score_column = "score"


    def evaluate_dataset(
        self, ds, batch_query, batch_passage, batch_score,
        **kwargs,
    ) -> dict[str, float | None]:
        raise NotImplementedError("For benchmark but not needed for ViLARMOR.")

    @staticmethod
    def l2_normalize_list(tensor_list):
        """
        L2-normalize each 2D tensor in the list along its last dimension.
        """
        normalized = []
        for tensor in tensor_list:
            # tensor should be 2D: (sequence_length, embedding_dim)
            # F.normalize(..., dim=-1) divides each row by its L2 norm
            # ensuring that each row in the embedding is unit-length.
            normalized.append(F.normalize(tensor, p=2, dim=-1))
        return normalized    

    def rank(
        self,
        batch_query: int,
        batch_passage: int,
        batch_score: int | None = None,
        dataloader_prebatch_query: int | None = None,
        dataloader_prebatch_passage: int | None = None,
    ) -> list[int]:
        """
        Get ranked list of relevant documents using psuedo queries.
        Returns a list of documents sorted by decending relevance.
        """
        # Load datasets
        ds_corpus = self.ds.corpus
        ds_queries = self.ds.queries

        # Get image data
        image_ids: list[int] = ds_corpus[self.corpus_id_column]

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )

        norm_query_embeddings = self.l2_normalize_list(query_embeddings)

        passage_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        norm_passage_embeddings = self.l2_normalize_list(passage_embeddings)

        # Get the similarity scores
        # lower score means more relevant
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=norm_query_embeddings,
            # ps=norm_passage_embeddings,
            ps=norm_query_embeddings,
            batch_size=batch_score,
            device="cuda",
        ) # not normalized

        print(scores)

        # Get the set relevant images 
        ranking = get_global_document_ranking([scores])

        return ranking


    def pseudo_relevance_judgement(
        self, ranking: list[int]
    ) -> dict[str, dict[str, float]]:
        """
        Create a relevance list of the ranked documents using LLM and pseudo queries
        """
        judge = ViLARMoRJudge()
        queries_df = self.ds.queries.to_pandas()
        corpus_df = self.ds.corpus.to_pandas()

        pqrel_list = []   # {"query-id": 1, "corpus-id": 473, "score": 1}

        for corpus_id in ranking:
            image = self.ds.get_image(corpus_df, corpus_id)

            for query_id in self.ds.queries[self.query_id_column]:
                query = self.ds.get_query(queries_df, query_id)
                judgment = judge.is_relevant(query, image)
                pqrel_list.append({
                    "query-id": query_id, 
                    "corpus-id": corpus_id, 
                    "score": judgment,})
        
        with open(os.path.join(self.ds.name,"pqrel.json"), "w") as file:
            json.dump(pqrel_list, file, indent=4)

        return pqrel_list

    def evaluate(self, pqrel_list: dict[str, dict[str, int]],  ranking: list[str]
                 ) -> dict[str, float]:
        """
        Compute the final ranking of NGDC@10 using the relevance qrel and the ranked
        output from the retrievers
        """
        ds_qrels = Dataset.from_dict(pqrel_list)['train']

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
            results=ranking,
            ignore_identical_ids=False,
        )

        return metrics


class ViLARMoRFull(BaseViDoReEvaluator):
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
        vision_retriever_list: ViLARMoRRetriever,
        vilarmor_ds: ViLARMoRDataset,
        num_corpus:int=None # subset of images for testing
    ):

        # Dataset column names
        self.corpus_id_column = "corpus-id"
        self.query_id_column = "query-id"
        self.query_column = "query"
        self.passage_column = "image"
        self.score_column = "score"


    def evaluate_dataset(
        self, ds, batch_query, batch_passage, batch_score,
        **kwargs,
    ) -> dict[str, float | None]:
        raise NotImplementedError("For benchmark but not needed for ViLARMOR.")

    @staticmethod
    def l2_normalize_list(tensor_list):
        """
        L2-normalize each 2D tensor in the list along its last dimension.
        """
        normalized = []
        for tensor in tensor_list:
            # tensor should be 2D: (sequence_length, embedding_dim)
            # F.normalize(..., dim=-1) divides each row by its L2 norm
            # ensuring that each row in the embedding is unit-length.
            normalized.append(F.normalize(tensor, p=2, dim=-1))
        return normalized    

    def rank(
        self,
        batch_query: int,
        batch_passage: int,
        batch_score: int | None = None,
        dataloader_prebatch_query: int | None = None,
        dataloader_prebatch_passage: int | None = None,
    ) -> list[int]:
        """
        Get ranked list of relevant documents using psuedo queries.
        Returns a list of documents sorted by decending relevance.
        """
        # Load datasets
        ds_corpus = self.ds.corpus
        ds_queries = self.ds.queries

        # Get image data
        image_ids: list[int] = ds_corpus[self.corpus_id_column]

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )

        norm_query_embeddings = self.l2_normalize_list(query_embeddings)

        passage_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        norm_passage_embeddings = self.l2_normalize_list(passage_embeddings)

        # Get the similarity scores
        # lower score means more relevant
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=norm_query_embeddings,
            # ps=norm_passage_embeddings,
            ps=norm_query_embeddings,
            batch_size=batch_score,
            device="cuda",
        ) # not normalized

        print(scores)

        # Get the set relevant images 
        ranking = get_global_document_ranking([scores])

        return ranking


    def pseudo_relevance_judgement(
        self, ranking: list[int]
    ) -> dict[str, dict[str, float]]:
        """
        Create a relevance list of the ranked documents using LLM and pseudo queries
        """
        judge = ViLARMoRJudge()
        queries_df = self.ds.queries.to_pandas()
        corpus_df = self.ds.corpus.to_pandas()

        pqrel_list = []   # {"query-id": 1, "corpus-id": 473, "score": 1}

        for corpus_id in ranking:
            image = self.ds.get_image(corpus_df, corpus_id)

            for query_id in self.ds.queries[self.query_id_column]:
                query = self.ds.get_query(queries_df, query_id)
                judgment = judge.is_relevant(query, image)
                pqrel_list.append({
                    "query-id": query_id, 
                    "corpus-id": corpus_id, 
                    "score": judgment,})
        
        with open(os.path.join(self.ds.name,"pqrel.json"), "w") as file:
            json.dump(pqrel_list, file, indent=4)

        return pqrel_list

    def evaluate(self, pqrel_list: dict[str, dict[str, int]],  ranking: list[str]
                 ) -> dict[str, float]:
        """
        Compute the final ranking of NGDC@10 using the relevance qrel and the ranked
        output from the retrievers
        """
        ds_qrels = Dataset.from_dict(pqrel_list)['train']

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
            results=ranking,
            ignore_identical_ids=False,
        )

        return metricsclass EvaluatorSet():
    
