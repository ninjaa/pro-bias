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
5. export SAMBANOVA_API_KEY=<your-key>
6. export WEAVE_PARALLELISM=1

Then you can run each eval by name

```
python evals/eval_make_text_more_casual.py
```
