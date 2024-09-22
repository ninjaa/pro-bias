# Pro-Bias: Better Human-Aligned Subjective LLM Evals that can Self-Optimize (wnb-hack-20240921)

## Project Overview

This project, developed for the [wnb-hack-20240921](https://wandb.me/judge), focuses on creating better human-aligned subjective LLM judges. It consists of four main components:

1. **ComparisonGEval**: An enhanced framework for more consistent and human-aligned subjective evaluations in LLM outputs.
2. **Generation of synthetic data sets**: for demonstrating the efficacy of ComparisonGEval.
3. **Example Comparison G Evals**: Showcasing ComparisonGEval's versatility across various subjective tasks.
4. **Automated Essay Grading Agent**: An iterative system that improves rubrics to align LLM evaluations with human raters.

### 1. ComparisonGEval

ComparisonGEval is an enhancement to the existing GEval framework, designed to address inconsistencies in subjective evaluations:

- Based off of the [original GEval](https://docs.confident-ai.com/docs/metrics-llm-evals) from Confident AI's DeepEval and [this paper](https://arxiv.org/abs/2303.16634)
- Uses a structured prompt inspired by Braintrust's approach [here](https://web.archive.org/web/20240907011400/https://www.braintrust.dev/docs/cookbook/recipes/EvaluatingChatAssistant#improve-scoring-with-a-custom-scorer)
- Implements a choice-based scoring system (A to H) instead of direct numeric scores, which was the flaw of the original GEval. LLMs are [notoriously bad at numeric scoring](https://www.nyckel.com/blog/calibrating-gpt-classifications/), and this was found to lead to systematic biases in evaluation. Choice-based scoring mitigates this problem and was found to more consistently align with human judgment. See G eval prompt and translation to scores [here](src/metrics/comparison_g_eval/template.py). The final score emitted maps largely to letter grades - 0, 0.2, 0.4, 0.6, 0.8, 1.0 are the possible outputs.
- Requires explanations for each evaluation, enhancing transparency
- Results in more stable, interpretable, and human-aligned evaluation metrics

### 2. Generation of Synthetic Data Sets

To demonstrate ComparisonGEval's efficacy, we've created synthetic datasets using a combination of techniques:

Mostly these datasets were created with Claude 3.5 Sonnet. It excels at creating eval outputs. In reality these tend not to be diverse enough to be representative of real world data, but tend to give a really strong signal and suffice for evaluation purposes. They are remarkably good. 

These synthetic datasets serve as a testbed for ComparisonGEval, allowing us to evaluate its performance across a spectrum of subjective tasks.

The synthetic datasets we created were:


1. **Formal Text**: We enumerated a number of sentences in extremely formal English that would serve as a good test of the model's ability to make formal text more casual and for ComparisonGEval's ability to grade the prompt efficacy in a way that is aligned to human judgment. Link to dataset [here](datasets/formal_text.csv)
2. **Conversation Names**: We generated a number of chat conversations between simulated humans and a Career Navigator assistant. In our ComparisonGEval, we created a prompt to generate a short name for the conversation, and then rated how well the name fit the conversation in a manner that was aligned to human judgment. Link to dataset [here](datasets/conversation_names.csv)
3. **Blockbuster Movie Idea Generation**: We generated a number of viable blockbuser movie ideas by seeding first Sonnet, and then O1-Preview with a list of movie ides from reddit. We then asked the model to rate the ideas using a ComparisonGEval, and we demonstrate how it rates the movie ideas in a manner presumably aligned to human judgement. Link to dataset [here](datasets/sonnet_movie_ideas.csv) and [here](datasets/o1_preview_movie_ideas.csv)



### 3. Example Comparison G Evals

We've developed several example evaluations to showcase ComparisonGEval's versatility:

1. **Text Casualization**: Evaluating the effectiveness of making formal text more casual. Eval [here](evals/eval_make_text_more_casual.py)
2. **Conversation Naming**: Evaluating the effectiveness of generating a short name for a conversation. Eval [here](evals/eval_conversation_naming.py)
3. **Bad Title Detection**: Identifying and rating the quality of potentially misleading or poor titles. Eval [here](evals/eval_bad_title_detection.py)
4. **Blockbuster Movie Idea Assessment**: Judging the potential and creativity of movie concepts. Eval [here](evals/eval_blockbuster_movie_idea.py)

These evals demonstrate ComparisonGEval's ability to handle diverse subjective tasks consistently and align with human judgment.

### 4. Automated Essay Grading Agent

Building on ComparisonGEval, we've developed an agent that:

- Analyzes scores from human raters
- Creates evaluation steps for ComparisonGEval to emulate the human rater's grading rubric
- Iteratively improves the rubric to achieve a target weighted kappa score
- Compares LLM output to human data, further refining the rubric

This agent demonstrates the practical application of ComparisonGEval in creating more human-aligned AI systems for subjective tasks like essay grading.

**It routinely gets Substantial Agreement with human raters on the representative samples we have created, and Moderate to Substantial Agreement on the evaluation dataset.**

#### Interpreting Weighted Cohen's Kappa

The interpretation scale:

- κw < 0: Less than chance agreement
- κw = 0: No agreement beyond chance
- 0 < κw ≤ 0.20: Slight agreement
- 0.21 < κw ≤ 0.40: Fair agreement
- 0.41 < κw ≤ 0.60: Moderate agreement
- 0.61 < κw ≤ 0.80: Substantial agreement
- 0.81 < κw ≤ 1.00: Almost perfect agreement

Example: A Weighted Kappa of 0.65 indicates substantial agreement between raters.

#### Advantages of Weighted Kappa Over Unweighted Kappa

1. Reflects the degree of disagreement
2. Reduces the impact of minor disagreements
3. Better for ordered categories

#### Limitations of Weighted Kappa

1. More complex to calculate and interpret
2. Choice of weights can influence the Kappa value
3. Sensitive to prevalence and bias, similar to unweighted Kappa

## Project Setup

### Installation

1. Clone the repo
2. Create virtualenv with python 3.11 and activate it
3. pip install -r requirements.txt
4. wandb login
5. export OPENAI_API_KEY=<your-key> OR set USE_SAMBANOVA=1 and SAMBANOVA_API_KEY=<your-key> to use Sambanova instead of OpenAI
6. export WEAVE_PARALLELISM=1
7. For the essay grader, export ANTHROPIC_API_KEY=<your-key>


## Running Evals & Reproducing Results

Use the `./run_python.sh` script to run any python script since it will fix up the PYTHONPATH.

 
### ENV Variables

NO_ASYNC_MODE=1 will run the eval in sync mode
USE_SAMBANOVA=1 will run the eval using Sambanova instead of OpenAI

### Running Migrations

To work with evaluations using Weights & Biases (W&B), you'll need to handle migrations to upload datasets and run evals that integrate with W&B. Below is a concise guide on how to achieve this.

#### Migrations: Uploading Datasets to Weights & Biases

Migrations are scripts used to upload datasets into Weave (W&B's data system). The migration scripts are located in the `migrations/` directory. Here's how you can use them:

#### Example Migration Script

Below is an example of a migration script that uploads the `sonnet_movie_ideas` dataset:

```python
def up():
    weave.init(config.WEAVE_PROJECT)

    df = pd.read_csv(FILE_PATH)
    dataset_rows = df.to_dict('records')

    dataset = Dataset(name='sonnet_movie_ideas', rows=dataset_rows)
    weave.publish(dataset)

    log.info(f"Uploaded {len(dataset_rows)} items to sonnet_movie_ideas dataset")
```

#### Migration command

```
./run_python.sh python migrations/upload_sonnet_movie_ideas.py
```

### Running Evals

Once you uploaded the relevant migrations, Then you can run each eval by name

```
NUM_EXAMPLES=10 ./run_python.sh python evals/eval_make_text_more_casual.py
```

or with Sambanova (because of rate limits, we don't use async mode & less examples)

```
NUM_EXAMPLES=5 NO_ASYNC_MODE=1 USE_SAMBANOVA=1 ./run_python.sh python evals/eval_make_text_more_casual.py
```

### Running the Essay Rubric Optimizer

To run the essay rubric optimizer, you can run the following command:
```
./run_python.sh python src/agents/essay_rubric_optimizer.py
```

When done it will publish a handy report in the reports folder with the final rubric and a plot of the kappa score at each iteration.

Example report [here](reports/optimization_report_20240922_121354.html)



## Todos

[x] make sure to tell the story of how we created the synthetic data
[ ] Look into W&B Feedback