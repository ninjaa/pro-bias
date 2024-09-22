import os
import openai
import re
import json
import time
from src.utils.run_essay_eval import load_dataset
import tiktoken

SYSTEM_PROMPT = """
You are an expert in rubric creation. Your goal is to improve the rubric.
"""

PROMPT_TEMPLATE = """
We are trying to create a set of evaluation criteria for grading essays.

The idea is to create a rubric that an LLM can use to evaluate an essay to match a human rater.

The following are the essays and the scores from a human rater.

{dataset}

so far we have created the following rubrics, and obtained the following weighted kappa scores against the human rater, as well as reasoning for why the scores are what they are against their own attempt rubric:

{rubric_history}

Your goal is to improve the rubric by creating a new rubric that is better. The order of the criteria within the rubric does matter, with the first criteria being the most important. Remember that your goal is to create a rubric that is as aligned as possible with the human rater, not to be the best rubric, but to be the best rubric that is aligned with the human rater. So try to roleplay as who the human rater might be and what they might be looking for.

You will have upto {max_iterations} iterations to create a new rubric. Feel free to backtrack if necessary to get the highest kappa score possible. In earlier runs, feel free to be quite creative in guessing what grade level the essay is for and how polished it is and what the human rater is looking for. If certain criteria should be ignored, feel free to mention that they should be ignored - for instance, in some cases it might be clear that punctuation should be ignored. In others, it might be clear that certain details have been redacted or anonymized, and perhaps the rubric should be adjusted for that.

Please return your rubric as a json array inside a code block. Feel free to reason about why you think the rubric is good or bad outside of the code block.

eg:

```json
["The essay is well-structured and organized.", "Does the essay contain a clear thesis or main idea?"]
```

Next Rubric:

"""


def improve_essay_rubric(rubric_history, rater_id, max_iterations):
    dataset = load_dataset("representative", rater_id)
    client = openai.OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            dataset=dataset,
            rubric_history=rubric_history,
            max_iterations=max_iterations)}
    ]

    # Count tokens using cl100k_base encoding
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = sum(len(encoding.encode(msg["content"])) for msg in messages)

    print(f"Debug - Total tokens in prompt: {num_tokens}")
    print("Debug - Messages array:")
    print(json.dumps(messages, indent=2))

    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=messages,
        temperature=0.7
    )

    response_content = response.choices[0].message.content

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
