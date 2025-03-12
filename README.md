# ViLARMoR (Vision Language Model Assisted Retriever Model Ranking)

This project was developed independently of the code base for LARMOR and as an extension of LARMOR.  It extends LARMOR from text only into vision-language domain.  Though it follows some of the same conventions.  

The original LARMOR research:
* code: https://github.com/ielab/larmor
* paper: https://arxiv.org/pdf/2402.04853

# Installation
Install the following dependencies before running the script.  Otherwise everything is handled by the script.

## Dependencies for ViDoRe Benchmark

* `conda create -n vidore python=3.13`
* `conda activate vilarmor`
* `conda install -c conda-forge sentencepiece`
* `pip install "vidore-benchmark[all-retrievers]"`
* `pip install autoawq`
* Install colpali-engine from source to include ColQwen2.5: `pip install git+https://github.com/illuin-tech/colpali`


# Hardware Requirements

VRAM above 65GB.  Tested on a single 80GB A100 or H100 NVIDIA GPU.

# Usage

Run the python file `run_vilarmor.py` to process the entire ViLARMOR pipeline including downloading necessary datasets.