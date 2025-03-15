"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""
import json
import os
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
from fusion import get_fusion_per_query

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
        model_conf: dict[str : tuple[PreTrainedModel, ProcessorMixin]],
        num_image_samples: int,
        num_pqueries: int,
        num_images_test: int = None,  # subset of images for testing
    ):
        super().__init__(vision_retriever=None)
        self.ds: ViLARMoRDataset = None
        self.vision_retriever: ViLARMoRRetriever = None
        self.ds_names = ds_names
        self.model_conf = model_conf
        self.num_images_test = num_images_test
        self.num_image_samples = num_image_samples
        self.num_pqueries = num_pqueries
        self.doc_ranking: dict[str, float] = {}
        self.dataset_pqrels: dict[str, dict[str, int]] = {}
        self.model_ndgc: dict[str, float] = {}

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

    def pseudo_relevance_judgement(self, judge: ViLARMoRJudge) -> dict[str, dict[str, int]]:
        """
        Create a relevance dictionary of the ranked documents using LLM and pseudo queries.
        """
        queries_df = self.ds.queries.to_pandas()
        corpus_df = self.ds.corpus.to_pandas()

        pqrels = defaultdict(dict)  # Change to dictionary format

        for query_id, docs in self.doc_ranking[self.ds.name].items():
            for corpus_id_key in docs.keys():  # Extract document IDs
                corpus_id = str(int(corpus_id_key))  # Ensure it's a string
                image = self.ds.get_image(corpus_df, corpus_id)
                query = self.ds.get_query(queries_df, query_id)
                judgment = judge.is_relevant(query, image)
                pqrels[str(query_id)][corpus_id] = judgment  # Store in correct format

        return pqrels



    def judge_all_datasets(self):
        """
        Judge all the datasets using reduced set of ranked documents (images)
        for each dataset and associated pseudo queries.
        """
        judge = ViLARMoRJudge()
        for dataset_name in self.ds_names:
            print(f"Generating pqrels for {dataset_name}")
            self.ds = ViLARMoRDataset(
                name=dataset_name, 
                num_images_test=None,  # need full set
                num_pqueries=None,
                num_image_samples=None
            )
            pqrels = self.pseudo_relevance_judgement(judge)
            self.dataset_pqrels[dataset_name] = pqrels
            with open(os.path.join(dataset_name, "pqrels.json"), "w") as file:
                json.dump(pqrels, file, indent=4)

        

    def score_all(self):
        # get the scores for each retriever and dataset pairing
        scores = {}
        for model_name in self.model_conf:
            model_class, processor_class = self.model_conf[model_name]
            self.vision_retriever = ViLARMoRRetriever(model_name, model_class, 
                                                        processor_class)
            scores[model_name] = {}
            for ds_name in self.ds_names:
                self.ds = ViLARMoRDataset(
                    name=ds_name, 
                    num_images_test=self.num_images_test,
                    num_pqueries=self.num_pqueries, 
                    num_image_samples=self.num_image_samples
                )
                score = self.score_single_model_corpus(
                    batch_query=BATCH_SIZE,
                    batch_image=BATCH_SIZE,
                    batch_score=BATCH_SIZE,
                )
                scores[model_name][ds_name] = score
            
        return scores

    def rank_all(self, scores):
        """
        Rank documents for each dataset using query-specific fusion instead of global aggregation.
        """
        for ds_name in self.ds_names:
            print(f"Making the document importance ranking for {ds_name}")
            self.ds = ViLARMoRDataset(
                name=ds_name, 
                num_images_test=self.num_images_test,
                num_pqueries=self.num_pqueries, 
                num_image_samples=self.num_image_samples
            )
            
            query_ids, image_ids = self.ds.get_query_image_ids()
            dataset_scores = []
            for model_name in self.model_conf:
                dataset_scores.append(scores[model_name][ds_name])

            # Perform query-specific fusion (instead of global ranking)
            self.doc_ranking[ds_name] = get_fusion_per_query(
                dataset_scores, query_ids, image_ids
            )

            with open(os.path.join(ds_name, "doc_ranking.json"), "w") as file:
                json.dump(self.doc_ranking[ds_name], file, indent=4)



    def evaluate(self) -> dict[str, float]:
        """
        Compute the final ranking of NDGC@10 using the relevance qrel and the ranked
        output from the retrievers.
        """
        for ds_name in self.dataset_pqrels:
            pqrels = self.dataset_pqrels[ds_name]
            self.model_ndgc[ds_name] = {}

            if ds_name not in self.doc_ranking:
                raise ValueError(f"Document ranking missing for dataset {ds_name}")

            for model_name in self.model_conf:
                print(f"Computing final nDCG@10 scores for {model_name}")
                metrics = self.compute_retrieval_scores(
                    qrels=pqrels,
                    results=self.doc_ranking[ds_name],  # Pass the ranked results
                    ignore_identical_ids=False,
                )

                ndcg_at_10 = metrics['ndcg_at_10']

                self.model_ndgc[ds_name][model_name] = ndcg_at_10
                
                with open(os.path.join(ds_name, "ndcg.json"), "w") as file:
                    json.dump(self.model_ndgc[ds_name], file, indent=4)

        return self.model_ndgc


    def run(self):
        print("Begin ViLARMoR Evaluation")
        scores = self.score_all()
        self.rank_all(scores)
        self.judge_all_datasets()
        self.evaluate()