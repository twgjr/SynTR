from vlm import BaseVLM


class ViLARMoRJudge(BaseVLM):
    def __init__(self):
        super().__init__(model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

    def is_relevant(self, query, image) -> bool:
        prompt = (
            "For the following question, judge whether it is"
            " ‘Highly Relevant’, ‘Somewhat Relevant’,or ‘Not Relevant’ to "
            f"the image. Question: {query}"
        )

        messages = [self.message_template(prompt, image)]
        response = self.response(messages)

        if "Highly Relevant" in response:
            return True
        return False
