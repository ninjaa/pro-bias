# Creating Better Human-Aligned Subjective LLM Judges (wnb-hack-20240921)

## Project Overview

This project, developed for the wnb-hack-20240921, focuses on creating better human-aligned subjective LLM judges. It consists of four main components:

1. ComparisonGEval: An enhanced framework for more consistent and human-aligned subjective evaluations in LLM outputs.
2. Generation of synthetic data sets for demonstrating the efficacy of ComparisonGEval.
3. Example Comparison G Evals: Showcasing ComparisonGEval's versatility across various subjective tasks.
4. Automated Essay Grading Agent: An iterative system that improves rubrics to align LLM evaluations with human raters.

### 1. ComparisonGEval

ComparisonGEval is an enhancement to the existing GEval framework, designed to address inconsistencies in subjective evaluations:

- Based off of the [original GEval](https://docs.confident-ai.com/docs/metrics-llm-evals) from Confident AI's DeepEval and [this paper](https://arxiv.org/abs/2303.16634)
- Uses a structured prompt inspired by Braintrust's approach [here](https://web.archive.org/web/20240907011400/https://www.braintrust.dev/docs/cookbook/recipes/EvaluatingChatAssistant#improve-scoring-with-a-custom-scorer)
- Implements a choice-based scoring system (A to H) instead of direct numeric scores. See prompt [here](src/metrics/comparison_g_eval/template.py)
- Requires explanations for each evaluation, enhancing transparency
- Results in more stable, interpretable, and human-aligned evaluation metrics

### 2. Generation of Synthetic Data Sets

To demonstrate ComparisonGEval's efficacy, we've created synthetic datasets using a combination of techniques:

- Leveraging smaller language models (e.g., 8B parameters) to generate diverse, imperfect outputs
- Implementing specific prompts to create varied quality responses
- Utilizing data augmentation techniques to expand dataset diversity

These synthetic datasets serve as a testbed for ComparisonGEval, allowing us to evaluate its performance across a spectrum of subjective tasks.

### 3. Example Comparison G Evals

We've developed several example evaluations to showcase ComparisonGEval's versatility:

1. **Conversation Naming**: Assessing the quality and relevance of conversation titles.
2. **Text Casualization**: Evaluating the effectiveness of making formal text more casual.
3. **Bad Title Detection**: Identifying and rating the quality of potentially misleading or poor titles.
4. **Blockbuster Movie Idea Assessment**: Judging the potential and creativity of movie concepts.

These evals demonstrate ComparisonGEval's ability to handle diverse subjective tasks consistently and align with human judgment.

### 4. Automated Essay Grading Agent

Building on ComparisonGEval, we've developed an agent that:

- Analyzes scores from human raters
- Creates evaluation steps for ComparisonGEval to emulate the human rater's grading rubric
- Iteratively improves the rubric to achieve a target weighted kappa score
- Compares LLM output to human data, further refining the rubric

This agent demonstrates the practical application of ComparisonGEval in creating more human-aligned AI systems for subjective tasks like essay grading.
## The Story of ComparisonGEval

ComparisonGEval was born out of the need for more consistent and human-aligned subjective evaluations in LLM outputs. Here's its evolution:

1. **Origin**: We started with the GEval from Confident AI, which allows evaluation of input vs output using a structured framework.

2. **Challenge**: The original GEval, while powerful, sometimes produced inconsistent results at high temperatures, with scores inexplicably bouncing around.

3. **Innovation**: ComparisonGEval addresses this by:
   - Using a more structured prompt inspired by Braintrust's approach.
   - Implementing a choice-based scoring system (A to H) instead of direct numeric scores.
   - Requiring explanations for each evaluation, enhancing transparency.

4. **Result**: A more stable, interpretable, and human-aligned evaluation metric.

## Demonstrating Efficacy

To showcase ComparisonGEval's effectiveness, we've created several evals:

1. Conversation Naming
2. Text Casualization
3. Bad Title Detection
4. Blockbuster Movie Idea Assessment

These evals demonstrate the versatility and consistency of ComparisonGEval across various subjective tasks.


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
7. For the essay grader, export ANTHRIC_API_KEY=<your-key>
 
Options

NO_ASYNC_MODE=1 will run the eval in sync mode
USE_SAMBANOVA=1 will run the eval using Sambanova instead of OpenAI

Then you can run each eval by name

```
NUM_EXAMPLES=10 ./run_python.sh python evals/eval_make_text_more_casual.py
```

or with Sambanova (because of rate limits, we don't use async mode & less examples)

```
NUM_EXAMPLES=5 NO_ASYNC_MODE=1 USE_SAMBANOVA=1 ./run_python.sh python evals/eval_make_text_more_casual.py
```

To run the essay rubric optimizer, you can run the following command:
```
./run_python.sh python src/agents/essay_rubric_optimizer.py


## Todos

[x] Get humans to annotate and compare to G Eval
[x] Change it from USE_OPENAI=1 to USE_SAMBANOVA=1
[x] Try to find a more mixed up dataset than what we have now where everything is uncritically amazing ... perhaps create movie ideas from a 8B model? Or ask the GEval to be very critical & opinionated?
[ ] make sure to tell the story of how we created the synthetic data
[x] switch eval optimizer to sonnet!!
[ ] Look into W&B Feedback