"""
This is based on vidore_evaluator_beir.py with modifications to use with
the LARMOR method.
"""

import json
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, ProcessorMixin
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

            with open(os.path.join(self.ds.name, "results.json"), "w") as file:
                json.dump(results, file, indent=4)

        return results

    def rank(self):
        """
        Aggregate document rankings for all models for given dataset.
        """
        scores = self.score()
        # convert the scores to the BeIR qrel format and save them to 
        # results.json
        results = self.scores_to_results(scores)

        query_ids = self.ds.query_ids()
        image_ids = self.ds.image_ids()

        if (len(scores)>1):
            print(f"Ranking all models for {self.ds.name}")
            ranking = get_fusion_per_query(
                scores, query_ids, image_ids
            )
        else:
            ranking = scores # no fusion

        return ranking, results
                

    def pseudo_relevance_judgement(
        self, judge: ViLARMoRJudge, top_m: int, ranking
    ) -> dict[str, dict[str, int]]:
        """
        Create pseudo query relevance judgments for the ranks lists made by 
        scoring the retriever models results using the pseudo queries.

        Not to be confused with the pseudo_qrels made directly from the dataset
        using the pseudo query generator, which only have relevance for the 
        documents used to generate queries.

        This judges the retrieved documents which may include other documents
        not used to generate the pseudo queries.
        """
        pqrels = []

        queries = ranking.keys()

        for query_id_key in tqdm(queries, desc="Query"):
            # get top_m documents for each query from ranking
            image_id_keys = ranking[query_id_key][:top_m]
            query_id = int(query_id_key)
            
            for corpus_id_key in image_id_keys:
                corpus_id = int(corpus_id_key)
                image = self.ds.get_image(corpus_id)
                query = self.ds.get_query(query_id)

                if judge.is_relevant(query, image):
                    pqrels.append({
                        "query-id": query_id,
                        "corpus-id": corpus_id,
                        "score": 1,
                    })

        return pqrels

    def judge(self, top_m: int):
        """
        Judges the relevance of the top_k documents for each pseudo query of 
        each dataset using the VLM model.
        """
        pqj_path = os.path.join(self.ds.name, "pseudo_qrels_judge.json")
        if not os.path.exists(pqj_path):
            judge = ViLARMoRJudge()
            print(f"Generating relevance judgements for {self.ds.name}")
            # judge the top_m documents for each pseudo query
            pseudo_qrels_judge = self.pseudo_relevance_judgement(judge, top_m)

            with open(pqj_path, "w") as file:
                json.dump(pseudo_qrels_judge, file, indent=4)


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

        with open(os.path.join(self.ds.name, "metrics.json"), "w") as file:
            json.dump(final_metrics, file, indent=4)
        
        return final_metrics


    def run_judge(self, ds_name, judge_top_m, gen_top_p, gen_temperature, gen_num_pqueries, 
            gen_corpus_sample_size):
        print(f"Begin ViLARMoR Evaluation of {ds_name}")
        self.generate_psuedos(ds_name, gen_top_p, gen_temperature, gen_num_pqueries, 
                        gen_corpus_sample_size)
        ranking, results = self.rank()
        self.judge(top_m=judge_top_m, ranking=ranking)
        self.evaluate(results)
        print(f"ViLARMoR Evaluation Complete for {ds_name}")

    def run_generate_not_judge(self, ds_name, gen_top_p, gen_temperature, gen_num_pqueries, 
            gen_corpus_sample_size):
        print(f"Begin ViLARMoR Evaluation of {ds_name}")
        self.generate_psuedos(ds_name, gen_top_p, gen_temperature, gen_num_pqueries, 
                        gen_corpus_sample_size)
        _, results = self.rank()
        self.evaluate(results)
        print(f"ViLARMoR Evaluation Complete for {ds_name}")

    def filter_from_split(self, dataset_split):
        """
        Filters the HuggingFace Datasets based on a dataset split containing query IDs
        and their positive/negative image IDs. Updates self.ds.corpus, self.ds.queries,
        and self.ds.qrels with the filtered versions.
        """
        print("Filtering dataset with the provided split.")
        filtered_corpus_ids = set()
        filtered_query_ids = set()
        filtered_qrels_set = set()

        for sample in dataset_split:
            query_id = sample['query-id']
            filtered_query_ids.add(query_id)

            for image_id in sample['positive_passages']:
                filtered_corpus_ids.add(image_id)
                filtered_qrels_set.add((query_id, image_id, 1))

            for image_id in sample['negative_passages']:
                filtered_corpus_ids.add(image_id)

        # Apply filtering using HuggingFace Dataset filter method
        self.ds.corpus = self.ds.corpus.filter(
            lambda example: example["corpus-id"] in filtered_corpus_ids)
        self.ds.queries = self.ds.queries.filter(
            lambda example: example["query-id"] in filtered_query_ids)
        self.ds.qrels = self.ds.qrels.filter(
            lambda example: (
                example["query-id"], example["corpus-id"]) in filtered_qrels_set)

# static non-class function
def compute_metrics(checkpoint_path: str, split_name: str = "validation"):
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from peft import PeftModel, PeftConfig
    from types import MethodType
    from datasets import load_dataset
    
    assert split_name in {"validation", "test"}, f"Unsupported split_name: {split_name}"
    assert os.path.isdir(checkpoint_path), f"Checkpoint path does not exist: {checkpoint_path}"

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load base model
    base_model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        device_map="auto",
        torch_dtype=torch.float16
    )

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # Wrap it with the LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # Patch the actual inner model's inner_forward
    original_inner_forward = model.base_model.model.inner_forward

    def safe_inner_forward(self, *args, **kwargs):
        if "labels" in kwargs:
            kwargs.pop("labels")
        return original_inner_forward(*args, **kwargs)

    model.base_model.model.inner_forward = MethodType(safe_inner_forward, model.base_model.model)
    ### end patch

    # Load processor
    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        use_fast=False
    )

    # Load the split (val/test)
    split_file = f"./beir_splits/{'val' if split_name == 'validation' else 'test'}.jsonl"
    dataset_split = load_dataset("json", data_files={split_name: split_file})[split_name]

    # Init evaluator and dataset
    evaluator = ViLARMoREvaluator(model_conf={"colqwen_finetuned": [model, processor]})
    evaluator.ds = ViLARMoRDataset(name="vidore/docvqa_test_subsampled_beir", load_pseudos=True, load_judgements=False)

    # Filter for relevant subset of data
    evaluator.filter_from_split(dataset_split)

    # Run evaluation
    _, results = evaluator.rank()
    evaluator.evaluate(results)

    metrics_path = os.path.join("vidore/docvqa_test_subsampled_beir", "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    return metrics["colqwen_finetuned"]
    
if __name__ == "__main__":
    # test filtering the dataset
    import json
    from vilarmor_dataset import ViLARMoRDataset

    # Load top 3 entries from the JSONL split
    split_path = "beir_splits/val.jsonl"
    with open(split_path, "r") as f:
        dataset_split = [json.loads(line) for _, line in zip(range(3), f)]

    # Replace with actual model + processor if available
    dummy_model_conf = {
        "dummy-model": (None, None)
    }

    # Create evaluator instance
    evaluator = ViLARMoREvaluator(model_conf=dummy_model_conf)

    # Load full dataset
    dataset_name = "vidore/docvqa_test_subsampled_beir"
    evaluator.ds = ViLARMoRDataset(name=dataset_name, load_pseudos=True, load_judgements=False)

    # Filter using the top 3 split examples
    evaluator.filter_from_split(dataset_split)

    # Print a quick summary to validate
    print("Filtered queries:", evaluator.ds.queries)
    print("Filtered corpus:", evaluator.ds.corpus)
    print("Filtered qrels:", evaluator.ds.qrels)