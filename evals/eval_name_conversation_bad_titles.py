import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.metrics.comparison_g_eval.comparison_g_eval_metric import ComparisonGEval
import weave
from weave import Evaluation
import asyncio
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


is_valid_conversation_name_metric = ComparisonGEval(
    verbose_mode=True,
    name="Is Valid Conversation Name",
    evaluation_steps=[
        "The output should be a short name for the conversation",
        "The short name should describe at least one main theme of the conversation specifically",
        "The name should not mention the user or use the word 'conversation'",
        "The name should be professional and less than 4 words",
        "If the name is not clear, it should be 'Untitled'",
        "The output should not include punctuation or special formatting"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT],
    **config.get_model_param(),
    **config.get_async_param()
)

weave.init(config.WEAVE_PROJECT)

dataset_ref = weave.ref('name_conversation').get()


@weave.op()
def function_to_evaluate(conversation_text: str):
    # Generate intentionally bad titles
    bad_titles = [
        "This is a very long conversation title that exceeds four words",
        "User's Conversation About Something",
        "Untitled Conversation",
        "Professional Discussion!",
        "Chat #1234",
        "Amazing Adventures in UX",
        "AI Careers in Oncology",
        "Blockchain for Astronauts",
    ]
    import random
    return {'conversation_text': conversation_text, 'generated_name': random.choice(bad_titles)}


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
    rows = dataset.rows[:config.NUM_EXAMPLES] if config.NUM_EXAMPLES > 0 else dataset.rows
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
