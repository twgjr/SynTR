# based on example from https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    return model


def load_processor(model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"):
    # min-max pixels determined over the vidore collections
    # max pixels can be further reduce to fit into smaller RAM GPU or faster processing
    # with accuracy tradeoff
    min_pixels = 149000
    max_pixels = int(67958100 / 64) # max reduced by factor of 64 due to GPU OOM @ 80GB
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=min_pixels, max_pixels=max_pixels
    )

    return processor


def message_template(text, image_url):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {"type": "text", "text": text},
        ],
    }

    return message


def response(
    model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor, messages: list
):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            top_k=0,  # must set this to allow top_p
            temperature=1
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


if __name__ == "__main__":
    from dataset import load_local_dataset

    model = load_model()
    processor = load_processor()

    corpus, queries, qrels = load_local_dataset("vidore/docvqa_test_subsampled_beir")

    messages = [message_template("What is this?", corpus[0]["image"])]
    response = response(model, processor, messages)
    print(response)
