"""
Base class to interface with large vision language models (VLM) for text
generation.
Based on example from https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.
Could be generalized to other VLM models but not needed for this project.
"""

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info
import torch


class BaseVLM:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"):
        self.model_name = model_name
        self.device = torch.device("cuda")
        self.processor = self._load_processor()
        self.model = self._load_model()

    def _load_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        return model

    def _load_processor(self, min_pixels=149000, max_pixels=1061845):
        # min-max pixels determined over the vidore collections
        # max pixels can be further reduce to fit into smaller RAM GPU or faster
        # processing with accuracy tradeoff
        processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

        return processor

    def message_template(self, text, image):
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        }

        return message

    def response(
        self,
        messages: list,
        top_p=None,
        temperature=None,
    ):
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            # Inference: Generation of the output
            if top_p == None:
                # deterministic output, no sampling
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None
                )
            else:
                # do nucleus sampling with top_p
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    top_p=top_p,
                    top_k=0,
                    temperature=temperature
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]  # only one message at a time
