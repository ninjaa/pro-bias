import sys
import os
# Add the parent directory of 'test' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepeval.sambanova_llm import sambanova_openai
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import asyncio
from weave import Evaluation
import weave
from src.prompts.make_text_more_casual import run_prompt
from src.metrics.comparison_g_eval.comparison_g_eval_metric import ComparisonGEval



# Get the USE_OPENAI environment variable
USE_OPENAI = os.environ.get('USE_OPENAI', '0') == '1'
ASYNC_MODE = os.environ.get('ASYNC_MODE', '0') == '1'
NUM_EXAMPLES = int(os.environ.get('NUM_EXAMPLES', '-1')
                   )  # -1 means all examples


# Conditionally set the model
model_param = {} if USE_OPENAI else {'model': sambanova_openai}
async_param = {'async_mode': ASYNC_MODE}

is_text_more_casual_metric = ComparisonGEval(
    verbose_mode=True,
    name="Is Text More Casual",
    evaluation_steps=[
         "Actual output should be more casual than the input",
         "Even a single word change or minor phrasing difference that leans towards informality is sufficient.",
         "The overall tone may remain largely formal, but any small shift towards casualness should be recognized as a positive.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT],
    **model_param,
    **async_param
)

weave.init('pro-bias')


# Retrieve the dataset
dataset_ref = weave.ref('formal_texts').get()


@weave.op()
def function_to_evaluate(input_text: str):
    return {'input_text': input_text, 'generated_text': run_prompt(input_text)}


@weave.op()
def match_score(expected_output: str, model_output: dict) -> dict:

    # Simple scoring based on whether the output is different from the input
    test_case = LLMTestCase(
        input=model_output['input_text'],
        actual_output=model_output['generated_text']
    )
    metric = is_text_more_casual_metric
    metric.measure(test_case)

    score = metric.score
    reason = metric.reason if score < 1.0 else ""

    return {'score': score, 'reason': reason, 'passed': score > 0.5}


def create_examples(dataset):
    examples = []
    rows = dataset.rows[:NUM_EXAMPLES] if NUM_EXAMPLES > 0 else dataset.rows
    for row in rows:
        example = {
            "input_text": row['Formal Text'],
            "expected_output": "UNUSED: Informal version of the text"
        }
        examples.append(example)
    return examples


# Usage
examples = create_examples(dataset_ref)

# Set up the evaluation
evaluation = Evaluation(
    dataset=examples,
    scorers=[match_score]
)

# Run the evaluation
weave.init('make-text-casual-eval')
asyncio.run(evaluation.evaluate(function_to_evaluate))
