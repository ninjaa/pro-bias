import os
import anthropic
import re
import json
import time
from src.utils.run_essay_eval import load_dataset
import tiktoken

SYSTEM_PROMPT = """
You are an AI who is expert in rubric creation and human alignment. Your goal is to improve the rubric so that LLM eval scores match those of the human rater highly.
"""

PROMPT_TEMPLATE = """
We're refining a rubric for essay grading that aligns with human rater assessments. Our goal is to improve agreement between LLM and human scores.

Dataset: {dataset}
Rubric History: {rubric_history}
Current Iteration: {current_iteration}/{max_iterations}
Current Rubric: {current_rubric}
Current Kappa: {current_kappa}
Target Kappa: {target_kappa}

Guidelines:
1. Analyze the current rubric and its performance.
2. Study the last run in the rubric_history. For each row, examine the ai_reasons to understand why there might be a delta between the target human_score and the ai_score.
3. Look at these reasons individually and in aggregate to identify patterns or common issues. Remember, you are not looking for reasons why the answer is wrong, you are looking for reasons why the current rubric is not matching that of a human rater whose scores have been collected. Therefore also look to the essay text for that essay for clues. The goal is ALIGNMENT, to minimize the delta between the ai_score and the human_score on grading the essay by fine-tuning the essay grading rubric. So look at failures where there is a larger delta than others and ask yourself why the ai_score is not matching the human_score for those cases. Is it something to do with the essay text that the LLM is not matching the human rater? Perhaps some detail is being graded for that the human rater was deliberately not grading for? For instance, depending on the context of how the essay was written, spellings, punctuation, paragraphs may not matter. Or perhaps the essay has had some personal information of names and places redacted? Or perhaps some grading criteria is missing?
4. Consider studying more successful rubric runs (by weighted kappa) in a similar manner.
5. Use a <Scratchpad> section to combine observations about what worked and what didn't work in different rubrics.
6. Based on these insights, identify key areas for improvement.
7. Propose concise, impactful changes.
8. Order criteria by importance to the human raters. Imagine what the human rater is looking for, and expand that into evaluation steps.
9. Keep the rubric flexible and adaptable.
10. Don't hesitate to simplify, remove, or reorder criteria if they are not useful.
11. If the weighted kappa is not improving significantly between iterations, get more creative and also feel free to slash and burn the existing rubric and start over with a new lightweight one.

How the rubric is evaluated:

The rubric is evaluated by the following code:
```python
   @staticmethod
    def generate_evaluation_results(evaluation_steps, text, parameters):
        # Prompt inspired by https://web.archive.org/web/20240907011400/https://www.braintrust.dev/docs/cookbook/recipes/EvaluatingChatAssistant#improve-scoring-with-a-custom-scorer
        return f\"\"\"Given the evaluation steps, assess the text and choose the most appropriate option from A to H, where:
A: Partially meets the criteria
B: Almost fully meets the criteria
C: Fully meets all criteria
D: Completely fails to meet the criteria
E: Successfully meets the criteria
F: Mostly fails to meet the criteria
G: Unrelated to the criteria
H: Slightly fails to meet the criteria

Evaluation Steps:
{{evaluation_steps}}

Text to evaluate:
{{text}}

**
IMPORTANT: Please return your response in JSON format with two keys: 'choice' and 'reason'. The 'choice' should be one of the options A-H, and the 'reason' should explain your selection without mentioning any numerical scores. DO NOT QUOTE THE SCORE in your reason.

Example JSON:
{{
    "choice": "B",
    "reason": "The text mostly meets the criteria outlined in the evaluation steps. It addresses the main points effectively, but lacks some minor details."
}}
**

JSON:
\"\"\"

    @staticmethod
    def calculate_score(choice):
        choice_scores = {{
            "A": 0.6,
            "B": 0.8,
            "C": 1,
            "D": 0,
            "E": 1,
            "F": 0.2,
            "G": 0,
            "H": 0.4
        }}
        return choice_scores.get(choice, 0)


```

<Scratchpad>
Analyze last run:
- Examine each row's ai_reasons for score discrepancies
- Identify patterns or common issues

Study successful rubrics:
- Compare successful rubrics (high weighted kappa)
- Note effective criteria and evaluation approaches

Combine observations:
- List what worked well across different rubrics
- Identify areas for improvement based on less successful rubrics
</Scratchpad>

Provide a brief explanation of your changes, then output the updated rubric as a JSON array in a code block:

Example output for a rubric where the human raters are apparently grading a conversation between two old people:

```json
[
    "Is this a short conversation among two old people?",
    "Does the conversation transcript feature at least one main theme?",
    "Is the conversation transcript engaging?",
    "Is the conversation transcript interesting?",
    "If there are redactions, ignore them",
    "Verbal tics like 'um' and 'ah' should be ignored"
]
```

Use clues from the dataset and analysis to infer what the human raters are looking for, and then expand that into evaluation steps.

"""


def improve_essay_rubric(rubric_history, rater_id, max_iterations, target_kappa):
    dataset = load_dataset("representative", rater_id)
    client = anthropic.Anthropic()
    current_iteration = rubric_history[-1]["iteration"]
    current_rubric = rubric_history[-1]["rubric"]
    current_kappa = rubric_history[-1]["kappa"]

    formatted_template = PROMPT_TEMPLATE.format(
        dataset=dataset,
        rubric_history=rubric_history,
        max_iterations=max_iterations,
        current_iteration=current_iteration,
        current_rubric=current_rubric,
        current_kappa=current_kappa,
        target_kappa=target_kappa
    )

    prompt = f"{SYSTEM_PROMPT}\n\n{formatted_template}"

    # Debug prints (optional)
    # print(f"Debug - Total characters in prompt: {len(prompt)}")
    # print("Debug - Prompt:")
    # print(prompt)

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_content = response.content[0].text

    max_retries = 3
    for attempt in range(max_retries):
        json_match = re.search(r'```json\n(.*?)\n```',
                               response_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
            try:
                rubric = json.loads(json_content)
                return rubric
            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}: Failed to parse JSON content")
        else:
            print(
                f"Attempt {attempt + 1}: No JSON content found in the response")

        if attempt < max_retries - 1:
            time.sleep(2)  # Wait for 2 seconds before retrying

    print("Max retries reached. Unable to get valid JSON content.")
    return None
