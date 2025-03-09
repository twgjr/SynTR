import os

QG_PROMPTS = {
    os.path.join("vidore", "docvqa_test_subsampled_beir"): {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {corpus_image},
            },
            {
                "type": "text",
                "text": "Generate a question that the following image can answer.",
            },
        ],
    },
}
