# Inspired originally by https://github.com/confident-ai/deepeval/blob/729d505c703cee93b70bf955ac0e261acf00288a/deepeval/metrics/g_eval/g_eval.py
class ComparisonGEvalTemplate:
    @staticmethod
    def generate_evaluation_steps(parameters, criteria):
        return f"""Given an evaluation criteria which outlines how you should judge the {parameters}, generate 3-4 concise evaluation steps based on the criteria below. You MUST make it clear how to evaluate {parameters} in relation to one another.

Evaluation Criteria:
{criteria}

**
IMPORTANT: Please make sure to only return in JSON format, with the "steps" key as a list of strings. No words or explanation is needed.
Example JSON:
{{
    "steps": <list_of_strings>
}}
**

JSON:
"""

    @staticmethod
    def generate_evaluation_results(evaluation_steps, text, parameters):
        # Prompt inspired by https://web.archive.org/web/20240907011400/https://www.braintrust.dev/docs/cookbook/recipes/EvaluatingChatAssistant#improve-scoring-with-a-custom-scorer
        return f"""Given the evaluation steps, assess the text and choose the most appropriate option from A to G, where:
A: Partially meets the criteria
B: Almost fully meets the criteria
C: Fully meets all criteria
D: Completely fails to meet the criteria
E: Successfully meets the criteria
F: Mostly fails to meet the criteria
G: Unrelated to the criteria

Evaluation Steps:
{evaluation_steps}

Text to evaluate:
{text}

**
IMPORTANT: Please return your response in JSON format with two keys: 'choice' and 'reason'. The 'choice' should be one of the options A-G, and the 'reason' should explain your selection without mentioning any numerical scores. DO NOT QUOTE THE SCORE in your reason.

Example JSON:
{{
    "choice": "B",
    "reason": "The text mostly meets the criteria outlined in the evaluation steps. It addresses the main points effectively, but lacks some minor details."
}}
**

JSON:
"""

    @staticmethod
    def calculate_score(choice):
        choice_scores = {"A": 0.6, "B": 0.8, "C": 1, "D": 0, "E": 1, "F": 0.2, "G": 0}
        return choice_scores.get(choice, 0)
