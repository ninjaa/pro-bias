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
5. export OPENAI_API_KEY=<your-key> OR set USE_SAMBANOVA=1 and SAMBANOVA_API_KEY=<your-key> to use Sambanova instead of OpenAI
6. export WEAVE_PARALLELISM=1

Options

NO_ASYNC_MODE=1 will run the eval in sync mode
USE_SAMBANOVA=1 will run the eval using Sambanova instead of OpenAI

Then you can run each eval by name

```
NUM_EXAMPLES=10 python evals/eval_make_text_more_casual.py
```

or with Sambanova (because of rate limits, we don't use async mode & less examples)

```
NUM_EXAMPLES=5 NO_ASYNC_MODE=1 USE_SAMBANOVA=1 python evals/eval_make_text_more_casual.py
```

## Todos

[ ] Look into W&B Feedback
[ ] Get humans to annotate and compare to G Eval
[x] Change it from USE_OPENAI=1 to USE_SAMBANOVA=1
[ ] Try to find a more mixed up dataset than what we have now where everything is uncritically amazing ... perhaps create movie ideas from a 8B model? Or ask the GEval to be very critical & opinionated?
[ ] make sure to tell the story of how we created the synthetic data
