# ViLARMoR (Vision Language Model Assisted Retriever Model Ranking)

This project was developed independently of the code base for LARMOR and as an extension of LARMOR.  It extends LARMOR from text only into vision-language domain.  Though it follows some of the same conventions.  

The original LARMOR research:
* code: https://github.com/ielab/larmor
* paper: https://arxiv.org/pdf/2402.04853

# Installation
Install the following dependencies before running the script.  Otherwise everything is handled by the script.

## Dependencies

* `conda create -n vidore python=3.13`
* `conda activate vilarmor`
* `conda install -c conda-forge sentencepiece`
* `pip install "vidore-benchmark[all-retrievers]"`
* `pip install autoawq`
* Install colpali-engine from source to include ColQwen2.5: `pip install git+https://github.com/illuin-tech/colpali`
* `pip install ranx`

# Hardware Requirements

VRAM above 65GB.  Tested on a single 80GB A100 or H100 NVIDIA GPU.

# Usage

Run the python file `run_vilarmor.py` to process the entire ViLARMOR pipeline including downloading necessary datasets.

1. Overview
ViLARMoR (Vision-Language Model Assisted Retriever Model Ranking) extends the original Large Language Model Assisted Retriever Model Ranking (LARMOR) framework to vision-language retrieval. While it follows some of the same conventions found in the LARMOR approach, this project was developed independently of the original LARMOR codebase and focuses on image + text retrieval tasks.

Original LARMOR code: https://github.com/ielab/larmor
Original LARMOR paper: arXiv:2402.04853
2. Installation & Dependencies
Create & Activate Conda Environment

bash
Copy
Edit
conda create -n vilarmor python=3.13
conda activate vilarmor
Install Basic Dependencies

bash
Copy
Edit
conda install -c conda-forge sentencepiece
pip install "vidore-benchmark[all-retrievers]"
pip install autoawq
pip install ranx
Install ColPali Engine

bash
Copy
Edit
pip install git+https://github.com/illuin-tech/colpali
This includes models such as ColQwen2.5.

3. Hardware Requirements
GPU Memory: At least 65GB of VRAM is recommended.
Tested Setup: A single NVIDIA A100 80GB or NVIDIA H100 80GB GPU.
Due to large model sizes and high-resolution images, lower-memory GPUs may cause out-of-memory errors unless you apply additional quantization or reduce dataset size.

4. Usage
Run the Main Pipeline

bash
Copy
Edit
python run_vilarmor.py
This script will:

Download or load the vision-language datasets as needed (e.g., from ViDoRe).
Generate pseudo queries for each sampled image.
Compute embeddings and perform retrieval.
Apply reciprocal rank fusion (RRF) or similar techniques.
Perform LLM-based judging for pseudo relevance scores.
Produce final evaluation metrics and JSON outputs.
Adjusting Parameters

You can modify dataset names, model paths, batch sizes, and other hyperparameters directly in the source files (e.g., run_vilarmor.py, evaluator.py) to tailor the experiment to your setup.
5. Notes & References
ViLARMoR is an extension from text-based to vision-language retrieval, retaining LARMOR’s core ideas:
Generating pseudo queries from the corpus.
Using multiple candidate retrievers to score documents.
Letting an LLM (in this case, a vision-language model) produce pseudo relevance labels.
The approach has been tested with subsets of DocVQA and TAT-DQA from the ViDoRe benchmark, but the pipeline is general and can handle other vision-language datasets.
For more details about the research motivation and methodology, please see the Phase 2 Project Report.
Thank you for using ViLARMoR! If you encounter issues or have questions about adapting the pipeline to new datasets, please feel free to reach out.