import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import asyncio
from weave import Evaluation
import weave
from src.metrics.comparison_g_eval.comparison_g_eval_metric import ComparisonGEval



is_blockbuster_material_metric = ComparisonGEval(
    verbose_mode=True,
    name="Is Blockbuster Material",
    evaluation_steps=[
        "The movie idea should have a broad audience appeal, capable of attracting large audiences.",
        "The premise should be unique or present a fresh take on a familiar theme, showcasing creativity.",
        "The concept should offer artistic depth, allowing for meaningful themes and messages.",
        "The story should be engaging and compelling, with strong potential for visual storytelling.",
        "The idea should be feasible to produce with current technology and resources.",
        "The plot should allow for dynamic character development and emotional resonance.",
        "The genre and premise should align with current market trends or have the potential to start new trends.",
        "The idea should have international appeal, increasing its potential for global success.",
        "The concept should be adaptable for various marketing strategies and merchandising opportunities.",
        "Overall, the movie idea should stand out as both commercially viable and artistically significant."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT],
    **config.get_model_param(),
    **config.get_async_param()
)

weave.init(config.WEAVE_PROJECT)

dataset_ref = weave.ref('sonnet_movie_ideas').get()


@weave.op()
def function_to_evaluate(row: dict):
    return {'input': f"Genre: {row['Genre']}\nPremise: {row['Premise']}", 'output': "This is a potential blockbuster movie idea."}


@weave.op()
def match_score(expected_output: str, model_output: dict) -> dict:
    test_case = LLMTestCase(
        input=model_output['input'],
        actual_output=model_output['output']
    )
    metric = is_blockbuster_material_metric
    metric.measure(test_case)

    score = metric.score
    reason = metric.reason if score < 1.0 else ""

    return {'score': score, 'reason': reason, 'passed': score > 0.5}


def create_examples(dataset):
    examples = []
    rows = dataset.rows[:config.NUM_EXAMPLES] if config.NUM_EXAMPLES > 0 else dataset.rows
    for row in rows:
        example = {
            "row": row,
            "expected_output": "UNUSED: Evaluation of blockbuster potential"
        }
        examples.append(example)
    return examples


examples = create_examples(dataset_ref)

evaluation = Evaluation(
    dataset=examples,
    scorers=[match_score]
)

asyncio.run(evaluation.evaluate(function_to_evaluate))
