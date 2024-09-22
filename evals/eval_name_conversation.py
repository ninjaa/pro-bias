import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepeval.sambanova_llm import sambanova_openai
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import asyncio
from weave import Evaluation
import weave
from src.prompts.name_conversation import run_prompt
from src.metrics.comparison_g_eval.comparison_g_eval_metric import ComparisonGEval

USE_OPENAI = os.environ.get('USE_OPENAI', '0') == '1'
ASYNC_MODE = os.environ.get('ASYNC_MODE', '0') == '1'
NUM_EXAMPLES = int(os.environ.get('NUM_EXAMPLES', '-1'))

model_param = {} if USE_OPENAI else {'model': sambanova_openai}
async_param = {'async_mode': ASYNC_MODE}

is_valid_conversation_name_metric = ComparisonGEval(
    verbose_mode=True,
    name="Is Valid Conversation Name",
    evaluation_steps=[
        "The output should be a short name for the conversation",
        "The name should not mention the user or use the word 'conversation'",
        "The name should be professional and less than 4 words",
        "If the name is not clear, it should be 'Untitled'",
        "The output should not include punctuation or special formatting"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT],
    **model_param,
    **async_param
)

weave.init('pro-bias')

dataset_ref = weave.ref('name_conversation').get()

@weave.op()
def function_to_evaluate(conversation_text: str):
    return {'conversation_text': conversation_text, 'generated_name': run_prompt(conversation_text)}

@weave.op()
def match_score(expected_output: str, model_output: dict) -> dict:
    test_case = LLMTestCase(
        input=model_output['conversation_text'],
        actual_output=model_output['generated_name']
    )
    metric = is_valid_conversation_name_metric
    metric.measure(test_case)

    score = metric.score
    reason = metric.reason if score < 1.0 else ""

    return {'score': score, 'reason': reason, 'passed': score > 0.5}

def create_examples(dataset):
    examples = []
    rows = dataset.rows[:NUM_EXAMPLES] if NUM_EXAMPLES > 0 else dataset.rows
    for row in rows:
        example = {
            "conversation_text": row['conversation_text'],
            "expected_output": row['expected_output']
        }
        examples.append(example)
    return examples

examples = create_examples(dataset_ref)

evaluation = Evaluation(
    dataset=examples,
    scorers=[match_score]
)

asyncio.run(evaluation.evaluate(function_to_evaluate))