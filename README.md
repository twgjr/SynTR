# Installation

* `conda create -n vilarmor python=3.13`
* `conda activate vilarmor`
* `conda install -c conda-forge sentencepiece`
* `pip install "vidore-benchmark[all-retrievers]"`
* Install colpali-engineer from source to include ColQwen2.5: `pip install git+https://github.com/illuin-tech/colpali`
* If you want to use quantized on gpu for faster inference: `pip install bitsandbytes`
* speed up QWEN VLM on cuda: `pip install flash_attn`
