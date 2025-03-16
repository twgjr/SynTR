# ViLARMoR (Vision Language Model Assisted Retriever Model Ranking)

## Overview
ViLARMoR (Vision-Language Model Assisted Retriever Model Ranking) extends the original Large Language Model Assisted Retriever Model Ranking (LARMOR) framework to vision-language retrieval. While it follows some of the same conventions found in the LARMOR approach, this project was developed independently of the original LARMOR codebase and focuses on image + text retrieval tasks.

Original LARMOR code: https://github.com/ielab/larmor
Original LARMOR paper: arXiv:2402.04853

## Installation & Dependencies
Create & Activate Conda Environment

`conda create -n vilarmor python=3.13
conda activate vilarmor`

Install Basic Dependencies

`conda install -c conda-forge sentencepiece`

`pip install "vidore-benchmark[all-retrievers]"`

`pip install autoawq`

`pip install ranx`

Install ColPali Engine:  `pip install git+https://github.com/illuin-tech/colpali`  This includes models such as ColQwen2.5.

#  Hardware Requirements
Tested Setup: A single NVIDIA A100 80GB or NVIDIA H100 80GB GPU.

Reducing the maximum image resolution greatly improves the memory requirement but sacrifices some model accuracy.

#  Usage
Run the Main Pipeline

`python run_vilarmor.py`

This script will:

Download or load the vision-language datasets as needed (e.g., from ViDoRe).

Generate pseudo queries for each sampled image.
Compute embeddings and perform retrieval.
Apply reciprocal rank fusion (RRF) or similar techniques.

Perform VLM-based judging for pseudo relevance scores.
Produce final evaluation metrics and JSON outputs.
Adjusting Parameters

You can modify dataset names, model paths, batch sizes, and other hyperparameters directly in the source files (e.g., run_vilarmor.py, evaluator.py) to tailor the experiment to your setup.

#  Notes & References
ViLARMoR is an extension from text-based to vision-language retrieval, retaining LARMOR’s core ideas:

* Generating pseudo queries from the corpus.

* Using multiple candidate retrievers to score documents.
Letting an LLM (in this case, a vision-language model) produce pseudo relevance labels.

* The approach has been tested with subsets of DocVQA and TAT-DQA from the ViDoRe benchmark, but the pipeline is general and can handle other vision-language datasets.

* For more details about the research motivation and methodology, please see the Phase 2 Project Report.
Thank you for using ViLARMoR! If you encounter issues or have questions about adapting the pipeline to new datasets, please feel free to reach out.