import torch

from vidore_benchmark.retrievers import VisionRetriever
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

MODELS = {
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0",
    # "Metric-AI/colqwen2.5-3b-multilingual",
    ## below model requires remote code trust, security risk
    ## "Metric-AI/ColQwenStella-2b-multilingual",
    # "tsystems/colqwen2-2b-v1.0",
    # "vidore/colqwen2.5-v0.2",
    # "vidore/colqwen2-v1.0",
    # "vidore/colqwen2.5-v0.1",
    # "vidore/colqwen2-v0.1",
    # "vido/re/colsmolvlm-v0.1",
    ## compatibility error
    ## "MrLight/dse-qwen2-2b-mrl-v1",
    # "vidore/colpali2-3b-pt-448",
    # "vidore/colSmol-500M",
    # "vidore/ColSmolVLM-Instruct-500M-base",
}


class ViLARMoRRetriever(VisionRetriever):
    def __init__(self, model_name):
        self.model_name = model_name
        model = self._get_model_instance(model_name)
        processor = self._get_processor_instance(model_name)
        super().__init__(model=model, processor=processor)

    @staticmethod
    def _get_processor_instance(model_name: dict[str, list]) -> ProcessorMixin:
        try:
            instance = ProcessorMixin.from_pretrained(model_name)
        except Exception as e:
            print(f"Problem creating processor: {e}")
        return instance

    @staticmethod
    def _get_model_instance(model_name: dict[str, list]) -> PreTrainedModel:
        try:
            instance = PreTrainedModel.from_pretrained(
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
