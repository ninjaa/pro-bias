import os

class Config:
    USE_SAMBANOVA = os.environ.get('USE_SAMBANOVA', '1') == '1'
    ASYNC_MODE = os.environ.get('ASYNC_MODE', '0') == '1'
    NUM_EXAMPLES = int(os.environ.get('NUM_EXAMPLES', '-1'))
    WEAVE_PROJECT = os.environ.get('WEAVE_PROJECT', 'pro-bias')

    @classmethod
    def get_model_param(cls):
        from src.deepeval.sambanova_llm import sambanova_openai
        return {'model': sambanova_openai} if cls.USE_SAMBANOVA else {}

    @classmethod
    def get_async_param(cls):
        return {'async_mode': cls.ASYNC_MODE}

config = Config()