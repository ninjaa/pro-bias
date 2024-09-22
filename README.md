# wnb-hack-20240921

## Goal

Update DeepEval with a Comparison G Eval

Run G Eval on several datasets, upload to W&B

Have humans annotate the results

Compare the human alignment to the Comparison G Eval

# Steps to run

1. Clone the repo
2. Create virtualenv with python 3.11 and activate it
3. pip install -r requirements.txt
4. wandb login
5. export SAMBANOVA_API_KEY=<your-key> OR set USE_OPENAI=1 to use OpenAI and make sure you have the OPENAI_API_KEY set
6. export WEAVE_PARALLELISM=1

Options

ASYNC_MODE=1 will run the eval in async mode
USE_OPENAI=1 will run the eval using OpenAI instead of Sambanova

Then you can run each eval by name

```
NUM_EXAMPLES=10 ASYNC_MODE=1 USE_OPENAI=1 python evals/eval_make_text_more_casual.py
```

## Todos

[ ] Look into W&B Feedback
[ ] Get humans to annotate and compare to G Eval

