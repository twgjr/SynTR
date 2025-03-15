"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""

from collections import defaultdict
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedModel, ProcessorMixin

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import (
    BaseViDoReEvaluator,
)
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever


from vilarmor_retriever import ViLARMoRRetriever
from judge import ViLARMoRJudge
from vilarmor_dataset import ViLARMoRDataset
from fusion import get_global_document_ranking

BATCH_SIZE = 1


class ViLARMoREvaluator(BaseViDoReEvaluator):
    """
    Applies the ViLARMoR evaluation method to evaluate the performance of a
    vision-and-language retrieval model. Designed to be used with the ViDoRe
    BeIR datasets format and v2 leaderboard.
    where each dataset contains 3 subsets:
        corpus: The dataset containing the corpus of documents (images).
        queries: The dataset containing the pseudo queries (text questions).
        qrels: The dataset containing the pseudo query relevance scores.
    """

    def __init__(
        self,
        ds_names: list[str],
        model_names: dict[str : tuple(PreTrainedModel, ProcessorMixin)],
        num_corpus: int = None,  # subset of images for testing
    ):
        super().__init__(vision_retriever=None)
        self.ds: ViLARMoRDataset = None
        self.vision_retriever: ViLARMoRRetriever = None
        self.ds_names = ds_names
        self.model_names = model_names
        self.num_corpus = num_corpus
        self.doc_ranking: dict[str, float] = []
        self.pqrels_map: dict[str, dict[str, int]] = []
        # pqrel_list format...
        # [
        #   dataset_name:
        #   [
        #       {"query-id", "corpus-id", "score"}
        #   ]
        # ]

        # Dataset column names
        self.corpus_id_column = "corpus-id"
        self.query_id_column = "query-id"
        self.query_column = "query"
        self.image_column = "image"
        self.score_column = "score"

    def evaluate_dataset(
        self, ds, batch_query, batch_passage, batch_score, **kwargs
    ) -> dict[str, float | None]:
        raise NotImplementedError("For benchmark but not needed for ViLARMOR.")

    @staticmethod
    def l2_normalize_list(tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
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

    def score_single_model_corpus(
        self,
        batch_query: int,
        batch_image: int,
        batch_score: int | None = None,
        dataloader_prebatch_query: int | None = None,
        dataloader_prebatch_image: int | None = None,
    ) -> torch.Tensor:
        """
        Get ranked list of relevant documents using psuedo queries.

        This creates a reduced set of ranked documents for each dataset
        representing the most relevant documents for the given pseudo queries,
        retriever model and dataset.

        Returns a 2D torch tensor of shape (queries, images).
        """
        # Load datasets
        ds_corpus = self.ds.corpus
        ds_queries = self.ds.queries

        # Get the embeddings for the queries and images
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )

        norm_query_embeddings = self.l2_normalize_list(query_embeddings)

        image_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.image_column,
            batch_passage=batch_image,
            dataloader_prebatch_size=dataloader_prebatch_image,
        )

        norm_image_embeddings = self.l2_normalize_list(image_embeddings)

        # Get the similarity scores
        # lower score means more relevant
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=norm_query_embeddings,
            ps=norm_image_embeddings,
            batch_size=batch_score,
            device="cuda",
        )  # not normalized

        return scores

    def pseudo_relevance_judgement(self, judge: ViLARMoRJudge) -> list[dict[str, int]]:
        """
        Create a relevance list of the ranked documents using LLM and pseudo queries
        """
        queries_df = self.ds.queries.to_pandas()
        corpus_df = self.ds.corpus.to_pandas()

        pqrel_list = []  # {"query-id": 1, "corpus-id": 473, "score": 1}

        for corpus_id, _ in self.doc_ranking[self.ds.name]:
            image = self.ds.get_image(corpus_df, corpus_id)

            for query_id in self.ds.queries[self.query_id_column]:
                query = self.ds.get_query(queries_df, query_id)
                judgment = judge.is_relevant(query, image)
                pqrel_list.append(
                    {
                        "query-id": query_id,
                        "corpus-id": corpus_id,
                        "score": judgment,
                    }
                )

        return pqrel_list

    def judge_all_datasets(self, judge: ViLARMoRJudge):
        """
        Judge all the datasets using reduced set of ranked documents (images)
        for each dataset and associated pseudo queries.
        """
        judge = ViLARMoRJudge()
        for dataset_name in self.ds_names:
            self.ds = ViLARMoRDataset(name=dataset_name, num_images=self.num_corpus)
            pqrels = self.pseudo_relevance_judgement(judge)
            self.pqrels_map[dataset_name] = pqrels

    def evaluate(
        self, pqrel_list: dict[str, dict[str, int]], ranking: list[str]
    ) -> dict[str, float]:
        """
        Compute the final ranking of NGDC@10 using the relevance qrel and the ranked
        output from the retrievers
        """
        ds_qrels = Dataset.from_dict(pqrel_list)["train"]

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

    def run(self):
        # get the scores for each retriever and dataset pairing
        scores = {}
        for model_name in self.model_names:
            self.vision_retriever = ViLARMoRRetriever(model_name)
            scores[model_name] = {}
            for ds_name in self.ds_names:
                self.ds = ViLARMoRDataset(name=ds_name, num_images=self.num_corpus)
                score = self.score_single_model_corpus(
                    batch_query=BATCH_SIZE,
                    batch_image=BATCH_SIZE,
                    batch_score=BATCH_SIZE,
                )
                scores[model_name][ds_name] = score

        # group scores by dataset, then make fused doc ranking
        for ds_name in self.ds_names:
            dataset_scores = []
            for model_name in self.model_names:
                dataset_scores.append(scores[model_name][ds_name])
             # get fused doc rank for dataset: list of (doc_id, score) tuples
            self.doc_ranking[ds_name] = get_global_document_ranking(dataset_scores)
