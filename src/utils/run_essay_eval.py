import argparse
import json
from typing import List, Optional
from pydantic import BaseModel
import random
import csv
from src.metrics.comparison_g_eval.comparison_g_eval_metric import ComparisonGEval
from src.config import config
from src.utils.safe_measure import safe_measure
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from sklearn.metrics import cohen_kappa_score
import numpy as np

class EssayEvalConfig(BaseModel):
    rubric: List[str]
    rater_id: int
    dataset: str
    num_examples: Optional[int] = None

def load_dataset(dataset: str, rater_id: int, num_examples: Optional[int] = None):
    if dataset == "representative":
        file_path = "datasets/aes/representative_samples.csv"
    else:
        file_path = "datasets/aes/mapped_rater_scores.csv"
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if num_examples:
        data = random.sample(data, min(num_examples, len(data)))
    
    return [{"essay_id": row['essay_id'], "essay_text": row['essay'], "score": float(row[f'rater{rater_id}_mapped_score'])} for row in data]

def run_essay_eval(eval_config: EssayEvalConfig):
    metric = ComparisonGEval(
        verbose_mode=True,
        name=f"Run Eval with Rubric (Rater {eval_config.rater_id})",
        evaluation_steps=eval_config.rubric,
        evaluation_params=[LLMTestCaseParams.INPUT,],
        **config.get_model_param(),
        **config.get_async_param()
    )

    dataset = load_dataset(eval_config.dataset, eval_config.rater_id, eval_config.num_examples)

    results = []
    human_scores = []
    ai_scores = []
    for item in dataset:
        test_case = LLMTestCase(
            input=item['essay_text'],
            actual_output=""  # We don't have a generated essay in this case
        )
        safe_measure(metric, test_case)
        results.append({
            'essay_id': item['essay_id'],
            'essay_text': item['essay_text'],
            'human_score': item['score'],
            'ai_score': metric.score,
            'ai_reason': metric.reason,
            'delta': metric.score - item['score'],
        })
        human_scores.append(item['score'])
        ai_scores.append(metric.score)

    # Calculate weighted kappa with scaled scores
    weighted_kappa = cohen_kappa_score(
        (np.array(human_scores) * 10).astype(int),
        (np.array(ai_scores) * 10).astype(int),
        weights='quadratic'
    )

    return results, weighted_kappa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run essay evaluation")
    parser.add_argument('eval_config', type=str, help='JSON configuration string')
    args = parser.parse_args()

    eval_config = EssayEvalConfig.model_validate_json(args.eval_config)
    results, weighted_kappa = run_essay_eval(eval_config)
    print(json.dumps({
        'results': results,
        'weighted_kappa': weighted_kappa
    }, indent=2))
