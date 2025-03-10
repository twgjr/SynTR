# Installation

## Dependencies for ViDoRe Benchmark

* `conda create -n vidore python=3.13`
* `conda activate vilarmor`
* `conda install -c conda-forge sentencepiece`
* `pip install "vidore-benchmark[all-retrievers]"`
* `pip install autoawq`
* Install colpali-engine from source to include ColQwen2.5: `pip install git+https://github.com/illuin-tech/colpali`

## Pseudo Query Generation and Relevance Judgements

* `conda create -n qwen python=3.13`
* `pip install  git+https://github.com/huggingface/transformers torchvision qwen-vl-utils`

# Requirements

* VRAM above 60GB.  Recommend running on 80GB A100 or H100
