from datasets import load_dataset, Dataset
import os
import json
from pseudo_query import PseudoQueryGenerator
from PIL import Image
import io

class ViLARMoRDataset:
    def __init__(self, name:str, corpus_size_limit:int=None, 
                 generator: PseudoQueryGenerator = None):
        self.name = name
        self.corpus: Dataset = None
        self.queries: Dataset = None
        self.qrels: Dataset = None

        if(os.path.exists(os.path.join(self.name, "corpus"))):
            self.load_corpus()

        if self.corpus is None:
            self.download_corpus(name, corpus_size_limit)

        if generator is not None:
            pseudo_queries, pseudo_qrel = generator.generate(
                dataset_name=name,
                corpus=self.corpus,
            )
            self.queries = pseudo_queries
            self.qrels = pseudo_qrel

            with open(os.path.join(name, "pseudo_queries.json"), "w") as f:
                json.dump(pseudo_queries, f, indent=4)

            with open(os.path.join(name, "pseudo_pqrels.json"), "w") as f:
                json.dump(pseudo_qrel, f, indent=4)
        else:
            self.load_queries()
            self.load_qrels()

        if self.queries is None or self.qrels is None:
            raise ValueError(
                "Queries or qrels not found.\n"
                "Provide a generator to generate pseudo queries and qrels.\n")
    
    def export_corpus_images_with_mapping(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        image_mapping = {}

        for item in self.corpus:
            corpus_id = str(item["corpus-id"])
            image = item["image"]

            if not isinstance(image, Image.Image):
                print(f"Warning: corpus_id {corpus_id} has invalid image object.")
                continue

            image_dir = os.path.join(output_dir, "image")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"{corpus_id}.png"
            image_path = os.path.join(image_dir, image_filename)

            try:
                image.save(image_path)
                image_mapping[corpus_id] = image_filename
            except Exception as e:
                print(f"Failed to save image {image_filename}: {e}")

        mapping_path = os.path.join(output_dir, "corpus_image_map.json")
        with open(mapping_path, "w") as f:
            json.dump(image_mapping, f, indent=4)

        print(f"Saved {len(image_mapping)} images and mapping to {output_dir}")

    def download_corpus(self, name, corpus_size_limit):
        try:
            corpus: Dataset = load_dataset(name, "corpus", split="test")
        except Exception as e:
            print(f"Failed to download dataset {name}: {e}")
            return

        if corpus_size_limit:
            corpus = corpus.select(range(corpus_size_limit))

        # Convert raw bytes to PIL.Image for every item
        corpus_data = []
        for item in corpus:
            image_obj = item["image"]

            if isinstance(image_obj, dict) and "bytes" in image_obj:
                img = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            elif isinstance(image_obj, Image.Image):
                img = image_obj.convert("RGB")
            else:
                print(f"Unrecognized image format for corpus-id {item['corpus-id']}, skipping.")
                continue

            corpus_data.append({
                "corpus-id": item["corpus-id"],
                "image": img
            })

        self.corpus = Dataset.from_list(corpus_data)

        # Save corpus as images + mapping
        corpus_dir = os.path.join(name, "corpus")
        self.export_corpus_images_with_mapping(corpus_dir)


    def load_corpus(self):
        """
        Loads the corpus from a directory containing corpus_image_map.json and image files.
        Reconstructs a HuggingFace Dataset object in-memory from the image paths.
        """
        corpus_dir = os.path.join(self.name, "corpus")
        mapping_path = os.path.join(corpus_dir, "corpus_image_map.json")

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Corpus mapping file not found at {mapping_path}")

        with open(mapping_path, "r") as f:
            image_mapping = json.load(f)

        corpus_data = []
        for corpus_id, filename in image_mapping.items():
            image_path = os.path.join(corpus_dir, "image", filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image file {filename} not found, skipping.")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
                continue

            corpus_data.append({
                "corpus-id": int(corpus_id),
                "image": image
            })


        # Rebuild the HuggingFace Dataset from list
        self.corpus = Dataset.from_list(corpus_data)


    def load_queries(self):
        pq_path = os.path.join(self.name, "pseudo_queries.json")
        queries = load_dataset("json", data_files=pq_path)
        self.queries = queries["train"]

    def load_qrels(self):
        pqrel_path = os.path.join(self.name, "pseudo_pqrels.json")
        qrels = load_dataset("json", data_files=pqrel_path)
        self.qrels = qrels["train"]

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

        if not isinstance(image, Image.Image):
            raise ValueError(f"Image for corpus_id {corpus_id} is not a valid PIL Image")

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
        query_ids = set()
        image_ids = set()
        for qrel in self.qrels:
            query_ids.add(qrel["query-id"])
            image_ids.add(qrel["corpus-id"])
        return list(query_ids), list(image_ids)
