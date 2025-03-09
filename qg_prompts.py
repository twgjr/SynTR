import os

QG_PROMPTS = {
    os.path.join("vidore", "docvqa_test_subsampled_beir"): {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "Generate a question that the following image can answer.",
            },
        ],
    },
}
