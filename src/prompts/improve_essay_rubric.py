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
3. Look at these reasons individually and in aggregate to identify patterns or common issues. Remember, you are not looking for reasons why the answer is wrong, you are looking for reasons why the current rubric is not matching that of a human rater whose scores have been collected. The goal is ALIGNMENT, to minimize the delta between the ai_score and the human_score by fine-tuning the rubric.
4. Consider studying more successful rubric runs (by weighted kappa) in a similar manner.
5. Use a <Scratchpad> section to combine observations about what worked and what didn't work in different rubrics.
6. Based on these insights, identify key areas for improvement.
7. Propose concise, impactful changes.
8. Order criteria by importance to the human raters. Imagine what the human rater is looking for, and expand that into evaluation steps.
9. Keep the rubric flexible and adaptable.
10. Don't hesitate to simplify, remove, or reorder criteria if they are not useful.
11. If the weighted kappa is not improving significantly between iterations, get more creative and also feel free to slash and burn the existing rubric and start over with a new lightweight one.

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
    print(f"Debug - Total characters in prompt: {len(prompt)}")
    print("Debug - Prompt:")
    print(prompt)

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
