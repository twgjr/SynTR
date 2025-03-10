from datasets import load_dataset, load_from_disk
import os
from PIL import Image
import io

dataset_names = [
    'vidore/docvqa_test_subsampled_beir',
    'vidore/tatdqa_test_beir',
]

def download_dataset(name):
    try:
        corpus = load_dataset(name, 'corpus', split='test')
        queries = load_dataset(name, 'queries', split='test')
        qrels = load_dataset(name, 'qrels', split='test')
    except Exception as e:
        print(f"Failed to download dataset {name}: {e}")

    # save to prefetch the data to speed up the evaluation
    corpus.save_to_disk(os.path.join(name, 'corpus'))
    queries.save_to_disk(os.path.join(name, 'queries'))
    qrels.save_to_disk(os.path.join(name, 'qrels'))

def load_local_dataset(name):
    corpus = load_from_disk(os.path.join(name, 'corpus'))
    queries = load_from_disk(os.path.join(name, 'queries'))
    qrels = load_from_disk(os.path.join(name, 'qrels'))
    return corpus, queries, qrels

# Function to extract area
def extract_area(image_binary):
    image = Image.open(io.BytesIO(image_binary["bytes"]))
    width, height = image.size
    return width * height  # Returning area as a single value

def find_max_min_area(image_corpus):
    # Convert dataset to pandas DataFrame
    df = image_corpus.to_pandas()

    # Compute area column
    df["area"] = df["image"].apply(lambda img: extract_area(img))

    # Get max and min areas
    max_area = df["area"].max()
    min_area = df["area"].min()

    return max_area, min_area

def find_image_range():
    areas = []
    for name in dataset_names:
        corpus, queries, qrels = load_local_dataset(name)
        areas.append(find_max_min_area(corpus))  # Ensure corpus is passed correctly

    # Extract max and min separately
    max_areas = [area[0] for area in areas]
    min_areas = [area[1] for area in areas]

    print(f'Max image pixels = {max(max_areas)}')
    print(f'Min image pixels = {min(min_areas)}')


if __name__ == '__main__':
    find_image_range()
    
    for name in dataset_names:
        if not os.path.exists(name):
            download_dataset(name)
        corpus, queries, qrels = load_local_dataset(name)
        if corpus and queries and qrels:
            print(corpus)
            print(queries)
            print(qrels)
            print('-----------------------------------')