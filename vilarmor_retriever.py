import torch

from vidore_benchmark.retrievers import VisionRetriever
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin




class ViLARMoRRetriever(VisionRetriever):
    def __init__(self, model_name):
        self.model_name = model_name
        model = self._get_model_instance(model_name)
        processor = self._get_processor_instance(model_name)
        super().__init__(model=model, processor=processor)

    @staticmethod
    def _get_processor_instance(model_name:str, processor_class:ProcessorMixin
                                ) -> ProcessorMixin:
        try:
            instance = processor_class.from_pretrained(model_name)
        except Exception as e:
            print(f"Problem creating processor: {e}")
        return instance

    @staticmethod
    def _get_model_instance(model_name:str, model_class:PreTrainedModel
                            ) -> PreTrainedModel:
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