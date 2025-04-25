"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""

import json
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, ProcessorMixin, set_seed
from datasets import load_dataset
from tqdm import tqdm


from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import (
    BaseViDoReEvaluator,
)
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from vidore_benchmark.retrievers import VisionRetriever

from vilarmor_retriever import ViLARMoRRetriever
from judge import ViLARMoRJudge
from vilarmor_dataset import ViLARMoRDataset
from fusion import get_fusion_per_query
from pseudo_query import PseudoQueryGenerator


set_seed(42)  # for consistent testing, sets all seeds for randomness
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
        model_conf: dict[str : tuple[PreTrainedModel, ProcessorMixin]],
    ):

        super().__init__(vision_retriever=None)
        self.ds: ViLARMoRDataset = None
        self.vision_retriever: ViLARMoRRetriever = None
        self.model_conf = model_conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Dataset column names
        self.corpus_id_column = "corpus-id"
        self.query_id_column = "query-id"
        self.query_column = "query"
        self.image_column = "image"
        self.score_column = "score"


    def generate_psuedos(self, name, top_p, temperature, num_pqueries, 
                        corpus_sample_size):
        """
        Downloads datasest and generates pseudo queries if needed
        """
        generator = PseudoQueryGenerator(top_p=top_p, temperature=temperature)
        self.ds = ViLARMoRDataset(name=name,load_pseudos=False, 
                                    load_judgements=False)
        pq_path = os.path.join(name, "pseudo_queries.json")
        pqrel_path = os.path.join(name, "pseudo_qrels.json")

        if not os.path.exists(pq_path) or not os.path.exists(pqrel_path):
            psuedo_queries, psuedo_qrels = generator.generate(
                                    dataset_name=name,
                                    corpus=self.ds.corpus,
                                    corpus_sample_size=corpus_sample_size,
                                    num_pqueries=num_pqueries
                                )

            with open(pq_path, "w") as f:
                json.dump(psuedo_queries, f, indent=4)

            with open(pqrel_path, "w") as f:
                json.dump(psuedo_qrels, f, indent=4)

        # reload the dataset with the generated pseudo queries and qrels
        self.ds = ViLARMoRDataset(name=name, load_pseudos=True, load_judgements = False)
        

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
        for idx, emb in enumerate(image_embeddings):
            if torch.isnan(emb).any():
                image_id = self.ds.image_ids()[idx]
                print(f"Image embedding for ID {image_id} is NaN. Replacing with zeros.")
                embedding_shape = emb.shape
                image_embeddings[idx] = torch.zeros_like(emb)

        norm_image_embeddings = self.l2_normalize_list(image_embeddings)

        # Get the similarity scores
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=norm_query_embeddings,
            ps=norm_image_embeddings,
            batch_size=batch_score,
            device="cuda",
        )

        return scores

    def score(self):
        # get the scores for each retriever model for the loaded dataset 
        scores = {}
        for model_name in self.model_conf:
            model, processor = self.model_conf[model_name]

            if isinstance(model, type) and isinstance(processor, type):
                # have class types not instances, load from huggingface
                model_class = model
                processor_class = processor
                self.vision_retriever = ViLARMoRRetriever(
                    model_name, model_class, processor_class
                )
            else:
                # assume have model and processor instances, directly use
                self.vision_retriever = VisionRetriever(
                    model=model, processor=processor)


            score = self.score_single_model_corpus(
                batch_query=BATCH_SIZE,
                batch_image=BATCH_SIZE,
                batch_score=BATCH_SIZE,
            )
            scores[model_name] = score

        return scores

    def scores_to_results(self, scores):
        """
        Converts the scores to the BeIR qrel format
        """
        results = {}

        query_ids = self.ds.query_ids()
        image_ids = self.ds.image_ids()

        for model_name in self.model_conf:
            results[model_name] = {}
            for query_idx, query_id in enumerate(query_ids):
                results[model_name][str(query_id)] = {}
                for img_idx, image_id in enumerate(image_ids):
                    results[model_name][str(query_id)][str(image_id)] = (
                        scores[model_name][query_idx][img_idx].item()
                    )
        return results

    def rank(self):
        """
        Aggregate document rankings for all models for given dataset.
        """
        scores = self.score()

        # convert the scores to the BeIR qrel format 
        results = self.scores_to_results(scores)

        query_ids = self.ds.query_ids()
        image_ids = self.ds.image_ids()

        if (len(scores)>1):
            print(f"Ranking all models for {self.ds.name}")
            ranking = get_fusion_per_query(
                scores, query_ids, image_ids
            )
        else:
            # convert scores of the single model to a ranking dict
            model_name = next(iter(scores))
            single_run = {str(query_ids[q_idx]): [
                            str(image_ids[i]) for i in torch.argsort(
                                -scores[model_name][q_idx]
                            )
                        ] for q_idx in range(len(query_ids))}
            ranking = single_run

        return ranking, results
                

    def pseudo_relevance_judgement(
        self,
        judge: ViLARMoRJudge,
        top_m: int,
        ranking,
        num_pos: int = 1,
        num_neg: int = 0,
        live_dump_path: str = None,
        final_path: str = None,
        dump_freq: int = 100,
    ) -> list[dict]:
        """
        Judges the relevance of top_k documents for each pseudo query.
        Resumes from live_dump_path if available.
        """
        pqrels = []
        
        queries = list(ranking.keys())

        # Resume from live file if exists
        if live_dump_path and os.path.exists(live_dump_path):
            with open(live_dump_path, "r") as f:
                pqrels = json.load(f)
            last_query_id = pqrels[-1]["query-id"]
            print(f"Resuming from live file: {live_dump_path} (last query-id = {last_query_id})")
            queries = queries[(last_query_id + 1):]

        for idx, query_id_key in enumerate(tqdm(queries, desc="Query")):
            image_id_keys = ranking[query_id_key][:top_m]
            query_id = int(query_id_key)
            print(f"Query: {query_id}, docs: \n{image_id_keys}")

            neg_count = 0
            pos_count = 0
            for corpus_id_key in image_id_keys:
                corpus_id = int(corpus_id_key)
                image = self.ds.get_image(corpus_id)
                query = self.ds.get_query(query_id)

                is_rel = judge.is_relevant(query, image)
                if is_rel is False:
                    pqrels.append({
                        "query-id": query_id,
                        "corpus-id": corpus_id,
                        "score": 0,
                    })
                    neg_count += 1
                elif is_rel is True:
                    pqrels.append({
                        "query-id": query_id,
                        "corpus-id": corpus_id,
                        "score": 1,
                    })
                    pos_count += 1

                if neg_count >= num_neg and pos_count >= num_pos:
                    break

            # Overwrite live file
            if live_dump_path and (idx % dump_freq == 0):
                with open(live_dump_path, "w") as f:
                    json.dump(pqrels, f, indent=4)
                print(f"Live dump updated at: {live_dump_path} [query {idx+1}/{len(queries)}]")

        # Final copy after full completion
        if final_path:
            with open(final_path, "w") as f:
                json.dump(pqrels, f, indent=4)
            print(f"Final pseudo qrels written to: {final_path}")

        return pqrels


    def judge(self, top_m: int, ranking):
        """
        Judges the relevance of the top_k documents for each pseudo query of 
        each dataset using the VLM model.
        """
        live_path = os.path.join(self.ds.name, "pseudo_qrels_judge_live.json")
        final_path = os.path.join(self.ds.name, "pseudo_qrels_judge.json")

        if not os.path.exists(final_path):
            judge = ViLARMoRJudge()
            print(f"Generating relevance judgements for {self.ds.name}")

            pseudo_qrels_judge = self.pseudo_relevance_judgement(
                judge, top_m, ranking,
                live_dump_path=live_path,
                final_path=final_path,
                dump_freq=100
    )

        # load the pseudo qrels with the relevance judgements
        self.ds = ViLARMoRDataset(
            name=self.ds.name,
            load_pseudos=True,
            load_judgements=True,
        )

    def evaluate(self, results) -> dict[str, float]:
        """
        Compute the final ranking of NDGC@10 using the qrels and the ranked
        output from the retrievers.
        """
        print(f"Computing metrics for {self.ds.name}")

        # Convert list to nested dictionary
        qrels = defaultdict(dict)
        for item in self.ds.qrels:
            qid = str(item["query-id"])
            cid = str(item["corpus-id"])
            score = item["score"]
            qrels[qid][cid] = score


        # Ensure all queries in results have entries in qrels
        # this may happen if the judge was more strict than the original query
        # generation
        for model_name in results:
            for query_id in results[model_name].keys():
                if query_id not in qrels:
                    qrels[query_id] = {}

        final_metrics = {}

        for model_name in self.model_conf:
            print(f"Computing metrics for {model_name}")
            print(f"len(qrels)={len(qrels)}, len(results)={len(results[model_name])}")

            metrics = self.compute_retrieval_scores(
                qrels=qrels,
                results=results[model_name],  # Pass the ranked results
                ignore_identical_ids=False,
            )

            final_metrics[model_name] = metrics

        return final_metrics

    def run_full(self, ds_name, judge_top_m, gen_top_p, gen_temperature, gen_num_pqueries, 
            gen_corpus_sample_size):
        print(f"Begin ViLARMoR Evaluation of {ds_name}")
        self.generate_psuedos(ds_name, gen_top_p, gen_temperature, gen_num_pqueries, 
                        gen_corpus_sample_size)
        ranking, results = self.rank()
        self.judge(top_m=judge_top_m, ranking=ranking)
        self.evaluate(results)

    def run_judge(self, ds_name, judge_top_m, queries_path, qrels_path):
        print(f"Begin Relevance Judgements of {ds_name}")
        self.ds = ViLARMoRDataset(name=ds_name, queries_path=queries_path, qrels_path=qrels_path)
        ranking, results = self.rank()
        with open("ranking_test.json", "w") as f:
            json.dump(ranking, f, indent=4)
        self.judge(top_m=judge_top_m, ranking=ranking)
        self.evaluate(results)

    def run_generate_not_judge(self, ds_name, gen_top_p, gen_temperature, gen_num_pqueries, 
            gen_corpus_sample_size):
        print(f"Begin Query Generation of {ds_name}")
        self.generate_psuedos(ds_name, gen_top_p, gen_temperature, gen_num_pqueries, 
                        gen_corpus_sample_size)
        _, results = self.rank()
        self.evaluate(results)



# static non-class function
def compute_metrics(model, processor, dataset:ViLARMoRDataset):
    model_conf={"model": [model, processor]}

    # Init evaluator and dataset
    evaluator = ViLARMoREvaluator(model_conf=model_conf)

    evaluator.ds = dataset

    # Run evaluation
    _, results = evaluator.rank()
    metrics = evaluator.evaluate(results)

    return metrics
    
if __name__ == "__main__":
    from colpali_engine.models import (
        ColQwen2_5,
        ColQwen2_5_Processor,
    )
    ds_name="vidore/docvqa_test_subsampled_beir"
    top_m=20
    evaluator = ViLARMoREvaluator(
        model_conf={
            "Metric-AI/colqwen2.5-3b-multilingual": [
                ColQwen2_5, ColQwen2_5_Processor]})
    evaluator.run_judge(ds_name,top_m,
        queries_path="pseudo_query_sets/general_judge-hard-3neg/pseudo_queries.json",
        qrels_path="vidore/docvqa_test_subsampled_beir/qrels.json" # does nothing for judging
    )