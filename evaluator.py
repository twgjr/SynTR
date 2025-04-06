"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""

import json
import os
from math import log10
from collections import defaultdict
import torch
import torch.nn.functional as F
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
from pseudo_query import PseudoQueryGenerator


BATCH_SIZE = 4


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
    ):

        super().__init__(vision_retriever=None)
        self.ds: ViLARMoRDataset = None
        self.vision_retriever: ViLARMoRRetriever = None
        self.ds_names = ds_names
        self.model_conf = model_conf
        self.doc_ranking: dict[str, float] = {}
        self.doc_importance_scores: dict[str, float] = {}
        self.pseudo_rel_list: dict[str, dict[str, int]] = {}
        self.model_ndgc: dict[str, float] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        print(f"len(ds.corpus)= {len(self.ds.corpus)}")
        print(f"len(ds.queries)= {len(self.ds.queries)}")

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

        print(f"Query embeddings shape: {norm_query_embeddings[0].shape}")
        print(f"Image embeddings shape: {norm_image_embeddings[0].shape}")
        print(f"Expected image IDs: {len(self.ds.corpus)}")  # Verify dataset size

        # Get the similarity scores
        # lower score means more relevant
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=norm_query_embeddings,
            ps=norm_image_embeddings,
            batch_size=batch_score,
            device="cuda",
        )  # not normalized

        return scores

    def pseudo_relevance_judgement(
        self, judge: ViLARMoRJudge, top_k: int
    ) -> dict[str, dict[str, int]]:
        """
        Create a relevance dictionary of the ranked documents using LLM and pseudo queries.
        """
        pqrels = defaultdict(dict)  # Change to dictionary format

        # slice the important documents
        slice_docs = dict(list(self.doc_importance_scores.items())[:top_k])

        for query_id in self.ds.queries[self.query_id_column]:
            for corpus_id_key in slice_docs:
                corpus_id = int(corpus_id_key)
                image = self.ds.get_image(corpus_id)
                query = self.ds.get_query(query_id)
                judgment = judge.is_relevant(query, image)
                print(f"q({query_id}) = query({query})")
                print(f"img({corpus_id}) = image({image})")
                print(f"q({query_id}), img({corpus_id}) = judgment({judgment})")
                pqrels[str(query_id)][corpus_id] = judgment  # Store in correct format

        return pqrels

    def judge_all_datasets(self, top_k: int, limit_corpus_size: int):
        """
        Judge all the datasets using reduced set of ranked documents (images)
        for each dataset and associated pseudo queries.

        top_k is the number of "important" documents to judge
        """
        judge = ViLARMoRJudge()
        for dataset_name in self.ds_names:
            print(f"Generating relevance list for {dataset_name}")
            self.ds = ViLARMoRDataset(
                name=dataset_name,
                generator=None,
                load_pseudos=True,
                limit_corpus_size=limit_corpus_size,
            )

            prl = self.pseudo_relevance_judgement(judge, top_k)
            self.pseudo_rel_list[dataset_name] = prl

        with open(os.path.join(dataset_name, "pseudo_rel_list.json"), "w") as file:
            json.dump(self.pseudo_rel_list, file, indent=4)

    def score_all(self, limit_corpus_size):
        # get the scores for each retriever and dataset pairing
        scores = {}
        for model_name in self.model_conf:
            model_class, processor_class = self.model_conf[model_name]
            self.vision_retriever = ViLARMoRRetriever(
                model_name, model_class, processor_class
            )

            scores[model_name] = {}
            for ds_name in self.ds_names:
                self.ds = ViLARMoRDataset(
                    name=ds_name,
                    generator=None,
                    load_pseudos=True,
                    limit_corpus_size=limit_corpus_size,
                )
                score = self.score_single_model_corpus(
                    batch_query=BATCH_SIZE,
                    batch_image=BATCH_SIZE,
                    batch_score=BATCH_SIZE,
                )
                scores[model_name][ds_name] = score
                print(f"scores[model_name][ds_name] shape = {score.shape}")

        return scores

    def scores_to_results(self, scores):
        """
        Converts the scores to the BeIR qrel format
        """
        results = {}

        for ds_name in self.ds_names:
            results[ds_name] = {}
            query_ids, image_ids = self.ds.get_query_image_ids()
            for model_name in self.model_conf:
                results[ds_name][model_name] = {}
                for query_idx, query_id in enumerate(query_ids):
                    results[ds_name][model_name][str(query_id)] = {}
                    for img_idx, image_id in enumerate(image_ids):
                        results[ds_name][model_name][str(query_id)][str(image_id)] = (
                            scores[model_name][ds_name][query_idx][img_idx].item()
                        )

                with open(os.path.join(ds_name, "results.json"), "w") as file:
                    json.dump(results[ds_name], file, indent=4)

        return results

    def rank_all(self, scores, limit_corpus_size):
        """
        Rank documents for each dataset using query-specific fusion instead of global aggregation.
        scores are indexed.  Need to be converted back to image and query id
        """
        for ds_name in self.ds_names:
            print(f"Making the document importance ranking for {ds_name}")

            self.ds = ViLARMoRDataset(
                name=ds_name,
                generator=None,
                load_pseudos=True,
                limit_corpus_size=limit_corpus_size,
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

    def doc_importance(self):
        """
        Fuse the query-image rankings by frequency of image id in ordered
        position for each query
        add to the important score of each document with inverse log of its position
        """
        doc_importance_scores = defaultdict(float)
        for ds_name in self.ds_names:
            for query_id in self.doc_ranking[ds_name]:
                for position, image_id in enumerate(
                    self.doc_ranking[ds_name][query_id]
                ):
                    discounted_score = 1 / (1 + log10(position + 1))  # Log discount
                    doc_importance_scores[
                        image_id
                    ] += discounted_score  # Accumulate score

        sorted_doc_importance_scores = dict(
            sorted(
                doc_importance_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        print(sorted_doc_importance_scores)

        self.doc_importance_scores = sorted_doc_importance_scores

        with open(os.path.join(ds_name, "doc_importance_scores.json"), "w") as file:
            json.dump(self.doc_importance_scores, file, indent=4)

    def evaluate(self, results) -> dict[str, float]:
        """
        Compute the final ranking of NDGC@10 using the relevance qrel and the ranked
        output from the retrievers.
        """
        for ds_name in self.pseudo_rel_list:
            p_rel_list = self.pseudo_rel_list[ds_name]
            self.model_ndgc[ds_name] = {}

            if ds_name not in results:
                raise ValueError(f"Document ranking missing for dataset {ds_name}")

            for model_name in self.model_conf:
                print(
                    f"Computing final nDCG@10 scores for {model_name} using: {results[ds_name][model_name]}"
                )
                metrics = self.compute_retrieval_scores(
                    qrels=p_rel_list,
                    results=results[ds_name][model_name],  # Pass the ranked results
                    ignore_identical_ids=False,
                )

                ndcg_at_10 = metrics["ndcg_at_10"]

                self.model_ndgc[ds_name][model_name] = ndcg_at_10

                with open(os.path.join(ds_name, "ndcg.json"), "w") as file:
                    json.dump(self.model_ndgc[ds_name], file, indent=4)

        return self.model_ndgc

    def init_datasets(self, top_p, temperature, num_pqueries, limit_corpus_size):
        """
        Downloads datasests and generates pseudo queries if needed
        """
        for name in self.ds_names:
            if not os.path.exists(name):
                generator = PseudoQueryGenerator(top_p, temperature, num_pqueries)
                ViLARMoRDataset(
                    name=name,
                    generator=generator,
                    load_pseudos=False,
                    limit_corpus_size=limit_corpus_size,
                )

    def run(self, top_k, top_p, temperature, num_pqueries, limit_corpus_size):
        print("Begin ViLARMoR Evaluation")
        self.init_datasets(top_p, temperature, num_pqueries, limit_corpus_size)
        scores = self.score_all()
        print(scores)
        results = self.scores_to_results(scores)
        print(results)
        self.rank_all(scores)
        self.doc_importance()
        self.judge_all_datasets(top_k=top_k)
        self.evaluate(results)
        # print("ViLARMoR Evaluation Complete")
