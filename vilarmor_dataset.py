from datasets import load_dataset, load_from_disk
import os
import json
from PIL import Image
import io
from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import (
    BEIRDataset,
)
from pseudo import PseudoQueryGenerator
from datasets import Dataset

COLLECTIONS = [
    "vidore/docvqa_test_subsampled_beir",
    "vidore/tatdqa_test_beir",
]


class ViLARMoRDataset:
    def __init__(self, name):
        self.name = name
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.qrels: Dataset = None

        if not os.path.exists(name):
            self.download_corpus()
            generator = PseudoQueryGenerator()
            generator.generate(name, self.corpus)
        else:
            self.load_local_dataset()

    def to_beir_dataset(self) -> BEIRDataset:
        return BEIRDataset(
            corpus=self.corpus,
            queries=self.queries,
            qrels=self.qrels,
        )

    def download_corpus(self):
        try:
            self.corpus = load_dataset(self.name, "corpus", split="test")
        except Exception as e:
            print(f"Failed to download dataset {self.name}: {e}")

        # save to prefetch the data to speed up the evaluation
        self.corpus.save_to_disk(os.path.join(self.name, "corpus"))

    def save_pseudos(self, psuedo_queries, pseudo_qrel):
        # Save to a JSON file
        with open(os.path.join(self.name, "pseudo_qrel.json"), "w") as f:
            json.dump(pseudo_qrel, f, indent=4)

        with open(os.path.join(self.name, "pseudo_queries.json"), "w") as f:
            json.dump(psuedo_queries, f, indent=4)

    def load_local_dataset(self):
        self.corpus = load_from_disk(os.path.join(self.name, "corpus"))
        pq_path = os.path.join(self.name, "pseudo_queries.json")
        self.queries = load_dataset("json", data_files=pq_path)
        pqrel_path = os.path.join(self.name, "pseudo_qrel.json")
        self.qrels = load_dataset("json", data_files=pqrel_path)

    # Function to extract area
    @staticmethod
    def _extract_area(image_binary):
        image = Image.open(io.BytesIO(image_binary["bytes"]))
        width, height = image.size
        return width * height  # Returning area as a single value

    def find_max_min_area(self):
        # Convert dataset to pandas DataFrame
        df = self.corpus.to_pandas()

        # Compute area column
        df["area"] = df["image"].apply(lambda img: self._extract_area(img))

        # Get max and min areas
        max_area = df["area"].max()
        min_area = df["area"].min()

        return max_area, min_area


def find_image_range():
    areas = []
    for name in COLLECTIONS:
        ds = ViLARMoRDataset(name, use_pseudo=False)
        areas.append(ds.find_max_min_area())

    # Extract max and min separately
    max_areas = [area[0] for area in areas]
    min_areas = [area[1] for area in areas]

    print(f"Max image pixels = {max(max_areas)}")
    print(f"Min image pixels = {min(min_areas)}")
