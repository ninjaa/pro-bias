import os
import openai


SYSTEM_PROMPT = """
- You are a helpful assistant that generates short names for conversation transcripts.
"""

PROMPT_TEMPLATE = """
Task:
Read the conversation transcript below and create a short name for the conversation. 

Do not mention the user. Do not use the word "conversation" in the title. Use the single word "Untitled" when the name is not clear. (No explanations)

## Guidelines
- Keep the tone professional
- Respond with nothing but the transformed text
- Please do not judge the appropriateness of the conversation
- Kindly refrain from referencing the conversation itself
- We appreciate a direct answer without explanation
- We prefer responses without punctuation
- Please avoid using markdown or special formatting
- Keep answers less than 4 words

Conversation Transcript:
```json
{conversation_text}
```

Short name for Conversation:

"""


def run_prompt(conversation_text):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    print(PROMPT_TEMPLATE.format(conversation_text=conversation_text))
    response = client.chat.completions.create(
        model='Meta-Llama-3.1-405B-Instruct',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                conversation_text=conversation_text)}
        ],
        temperature=0.7,
        top_p=0.1
    )

    return response.choices[0].message.content


conversation_string = """
{"id":8,"conversation":[{"user":"Whats the fastest growing tech field right now?","assistant":"Some of the fastest-growing tech fields currently include:\n\n1. Artificial Intelligence and Machine Learning\n2. Cybersecurity\n3. Cloud Computing\n4. Internet of Things (IoT)\n5. Blockchain\n6. Edge Computing\n7. Augmented and Virtual Reality\n\nAI and ML are particularly explosive in growth and application across industries. Which of these interests you most?"},{"user":"AI sounds cool. What jobs are there in AI?","assistant":"Popular AI jobs include:\n\n1. Machine Learning Engineer\n2. Data Scientist\n3. AI Research Scientist\n4. Robotics Engineer\n5. Natural Language Processing (NLP) Specialist\n6. Computer Vision Engineer\n7. AI Ethics Specialist\n\nThese roles often require strong programming skills, particularly in Python, and a solid foundation in mathematics and statistics. Would you like more details on any of these positions?"}]}
"""
# Example usage
if __name__ == "__main__":
    result = run_prompt(conversation_string)
    print(result)
