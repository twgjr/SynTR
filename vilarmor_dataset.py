from datasets import load_dataset, Dataset
import os
import json
from PIL import Image
from tqdm import tqdm


class ViLARMoRDataset:
    def __init__(
        self,
        name: str,
        load_pseudos: bool,
    ):
        self.name = name
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.qrels: Dataset = None

        # download the full dataset if it doesn't exist
        if not os.path.exists(self.name):
            corpus_data = self._download_corpus()
            queries_data, qrels_data = self._download_queries_qrels()
            os.makedirs(self.name, exist_ok=False)
            corpus_dir = os.path.join(self.name, "corpus")
            images_dir = os.path.join(corpus_dir, "images")
            os.makedirs(corpus_dir, exist_ok=False)
            os.makedirs(images_dir, exist_ok=False)
            corpus_id_image_list = []

            print(f"Saving images for {self.name}")
            tqdm_images = tqdm(
                corpus_data, desc=f"Saving", total=len(corpus_data)
            )
            for item in tqdm_images:
                item["image-obj"].save(os.path.join(images_dir, f"{item["corpus-id"]}.png"))
                corpus_id_image_list.append(
                    {"corpus-id":item["corpus-id"], 
                     "image-path":item["image-path"]})

            image_map_path = os.path.join(corpus_dir, "corpus_id_image_map.json")
            self._save_data(corpus_id_image_list, image_map_path)
            self._save_data(queries_data, os.path.join(self.name, "queries.json"))
            self._save_data(qrels_data, os.path.join(self.name, "qrels.json"))
            
        if load_pseudos:
            self._load_pseudos()
        else:
            self._load_trues()

        self.corpus = self._load_corpus_from(None)

    def _download_corpus(self):
        corpus: Dataset = load_dataset(self.name, "corpus")["test"]

        corpus_dir = os.path.join(self.name, "corpus")
        images_dir = os.path.join(corpus_dir, "images")

        corpus_data = []

        print(f"Downloading images for {self.name}")
        tqdm_corpus = tqdm(
            corpus, desc=f"Downloading", total=len(corpus)
        )
        for item in tqdm_corpus:
            image_obj = item["image"]
            image_id = item["corpus-id"]
            image_filename = f"{image_id}.png"
            image_path = os.path.join(images_dir, image_filename)
            corpus_data.append(
                {
                    "corpus-id": image_id,
                    "image-path": image_path,
                    "image-obj": image_obj,
                }
            )

        return corpus_data

    @staticmethod
    def _save_data(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def _download_queries_qrels(self):
        queries: Dataset = load_dataset(self.name, "queries")["test"]
        qrels: Dataset = load_dataset(self.name, "qrels")["test"]

        queries_data = []
        qrels_data = []

        for item in queries:
            queries_data.append({"query-id": item["query-id"], "query": item["query"]})

        for item in qrels:
            qrels_data.append(
                {
                    "query-id": item["query-id"],
                    "corpus-id": item["corpus-id"],
                    "score": item["score"],
                }
            )

        return queries_data, qrels_data

    def _load_corpus_from(self, image_ids: list[int]):
        corpus_dir = os.path.join(self.name, "corpus")
        mapping_path = os.path.join(corpus_dir, "corpus_id_image_map.json")
        corpus_data = []

        with open(mapping_path, "r") as f:
            image_mapping = json.load(f)

        corpus_data = []

        def add_corpus_data(corpus_id, image_path):
            image = Image.open(image_path)
            corpus_item = {"corpus-id":corpus_id, "image":image}
            corpus_data.append(corpus_item)

        if image_ids:
            for corpus_id in image_ids:
                image_path = image_mapping[corpus_id]['image-path']
                add_corpus_data(corpus_id, image_path)
        else:
            for item in image_mapping:
                corpus_id = item['corpus-id']
                image_path = image_mapping[corpus_id]['image-path']
                add_corpus_data(corpus_id, image_path)

        return Dataset.from_list(corpus_data)

    def _load_queries_qrels(self, querys_path: str, qrels_path: str):
        with open(qrels_path, "r") as f:
            qrels = json.load(f)

        with open(querys_path, "r") as f:
            queries = json.load(f)

        return Dataset.from_list(queries), Dataset.from_list(qrels)

    def _load_pseudos(self):
        pq_path = os.path.join(self.name, "pseudo_queries.json")
        pqrel_path = os.path.join(self.name, "pseudo_qrels.json")

        self.queries, self.qrels = self._load_queries_qrels(
            querys_path=pq_path,
            qrels_path=pqrel_path,
        )

    def _load_trues(self):
        queries_path = os.path.join(self.name, "queries.json")
        qrels_path = os.path.join(self.name, "qrels.json")

        self.queries, self.qrels = self._load_queries_qrels(
            querys_path=queries_path,
            qrels_path=qrels_path,
        )

    def get_image(self, corpus_id: int):
        # Filter the dataset to find items with matching corpus-id
        filtered_images = []
        for item in self.corpus:
            if int(item["corpus-id"]) == corpus_id:
                filtered_images.append(item)

        if not filtered_images:
            raise ValueError(f"No image found for corpus_id: {corpus_id}\n")

        if len(filtered_images) > 1:
            raise ValueError(f"Duplicate corpus_id found: {corpus_id}\n")

        image = filtered_images[0]["image"]

        return image

    def get_query(self, query_id: int):
        # Filter the dataset to find items with matching query-id
        filtered_queries = []
        for item in self.queries:
            if int(item["query-id"]) == query_id:
                filtered_queries.append(item)

        if not filtered_queries:
            raise ValueError(f"No query found for query_id: {query_id}\n")

        if len(filtered_queries) > 1:
            raise ValueError(f"Duplicate query_id found: {query_id}\n")

        return filtered_queries[0]["query"]

    def get_query_image_ids(self):
        """
        Get the query and image IDs from the qrels.  If using pseudo queries,
        the image IDs will be a smaller subset of the corpus IDs from the
        original dataset.
        """
        query_ids = set()
        image_ids = set()
        for qrel in self.qrels:
            query_ids.add(qrel["query-id"])
            image_ids.add(qrel["corpus-id"])
        return list(query_ids), list(image_ids)


if __name__ == "__main__":
    # Test without generator
    dataset = ViLARMoRDataset(
        name="vidore/docvqa_test_subsampled_beir",
        generator=None,
        load_pseudos=False,
    )
    print("Corpus:", dataset.corpus)
    print("Queries:", dataset.queries)
    print("Qrels:", dataset.qrels)
    print("Image IDs:", dataset.get_query_image_ids())
    image = dataset.get_image(4)
    image.save("test_image_4.png")
    print("Image:", dataset.get_image(4))
    print("Query:", dataset.get_query(1))
