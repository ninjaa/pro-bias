# from langchain_openai import ChatOpenAI
# from deepeval.models.base_model import DeepEvalBaseLLM
# import os


from deepeval.models.base_model import DeepEvalBaseLLM
import openai
import os


class SambanovaOpenAI(DeepEvalBaseLLM):
    def __init__(self, model_name: str = 'Meta-Llama-3.1-405B-Instruct'):
        super().__init__(model_name)
        self.client = self.load_model()

    def load_model(self):
        return openai.OpenAI(
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.1
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.1
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model_name


# Usage
sambanova_openai = SambanovaOpenAI()
print(sambanova_openai.generate("Write me a joke"))

# # Replace these with real values
# custom_model = ChatOpenAI(
#     api_key=os.environ.get("SAMBANOVA_API_KEY"),
#     base_url="https://api.sambanova.ai/v1",
#     model='Meta-Llama-3.1-405B-Instruct',
# )
# sambanova_openai = SambanovaOpenAI(model=custom_model)
# print(sambanova_openai.generate("Write me a joke"))
