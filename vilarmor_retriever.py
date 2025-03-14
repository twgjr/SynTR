import torch
from colpali_engine.models import (
    ColQwen2,
    ColQwen2Processor,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColIdefics3,
    ColIdefics3Processor,
    ColPali,
    ColPaliProcessor,
)

from vidore_benchmark.retrievers import VisionRetriever
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

MODELS = {
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0": [ColQwen2_5, ColQwen2_5_Processor],
    # "Metric-AI/colqwen2.5-3b-multilingual": [ColQwen2_5, ColQwen2_5_Processor],
    ## below model requires remote code trust, security risk
    ## "Metric-AI/ColQwenStella-2b-multilingual":[AutoModel, AutoProcessor],
    # "tsystems/colqwen2-2b-v1.0": [ColQwen2, ColQwen2Processor],
    # "vidore/colqwen2.5-v0.2": [ColQwen2_5, ColQwen2_5_Processor],
    # "vidore/colqwen2-v1.0": [ColQwen2, ColQwen2Processor],
    # "vidore/colqwen2.5-v0.1": [ColQwen2_5, ColQwen2_5_Processor],
    # "vidore/colqwen2-v0.1": [ColQwen2, ColQwen2Processor],
    # "vido/re/colsmolvlm-v0.1": [ColIdefics3, ColIdefics3Processor],
    ## compatibility error
    ## "MrLight/dse-qwen2-2b-mrl-v1": [AutoProcessor, Qwen2VLForConditionalGeneration],
    # "vidore/colpali2-3b-pt-448": [ColPali, ColPaliProcessor],
    # "vidore/colSmol-500M": [ColIdefics3, ColIdefics3Processor],
    # "vidore/ColSmolVLM-Instruct-500M-base": [ColIdefics3, ColIdefics3Processor],
}


class ViLARMoRRetriever(VisionRetriever):
    def __init__(self, model_name):
        self.model_name = model_name
        model = self._get_model_instance(model_name)
        processor = self._get_processor_instance(model_name)
        super().__init__(model=model, processor=processor)

    @staticmethod
    def _get_processor_instance(model_name: dict[str, list]) -> ProcessorMixin:
        processor_class: ProcessorMixin = MODELS[model_name][1]
        try:
            instance = processor_class.from_pretrained(model_name)
        except Exception as e:
            print(f"Problem creating processor: {e}")
        return instance

    @staticmethod
    def _get_model_instance(model_name: dict[str, list]) -> PreTrainedModel:
        model_class: PreTrainedModel = MODELS[model_name][0]

        try:
            instance = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            return instance
        except Exception as e:
            print(f"Problem with getting pretrained model: {e}")
            # rethrow
            raise e

    def init_retriever_instance(
        self, model: PreTrainedModel, processor: ProcessorMixin
    ) -> VisionRetriever:
        try:
            instance = VisionRetriever(model=model, processor=processor)
        except Exception as e:
            print(f"Problem with creating Retriever: {e}")
        return instance
