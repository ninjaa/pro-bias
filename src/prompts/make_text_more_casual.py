import os
import openai


SYSTEM_PROMPT = """
You are an expert content editor. Your goal is to make text more casual.

## Guidelines
- Make the text more causual
- Keep the tone professional
- Respond with nothing but the transformed text
"""

PROMPT_TEMPLATE = """
You are a helpful assistant that makes text more casual.

## Guidelines
- Make the text more causual
- Keep the tone professional
- Respond with nothing but the transformed text

Input:
{input_text}

Output:

"""


def run_prompt(input_text):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    response = client.chat.completions.create(
        model='Meta-Llama-3.1-405B-Instruct',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                input_text=input_text)}
        ],
        temperature=0.7,
        top_p=0.1
    )

    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    result = run_prompt("In sooth I know now why I am so sad")
    print(result)
