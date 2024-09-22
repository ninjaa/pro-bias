import json
from src.utils.run_essay_eval import run_essay_eval, EssayEvalConfig
from src.prompts.improve_essay_rubric import improve_essay_rubric


def optimize_rubric(initial_rubric, rater_id, target_kappa, max_iterations=10):
    rubric_history = []
    iteration = 0
    current_rubric = initial_rubric

    while iteration < max_iterations:
        config = EssayEvalConfig(
            rubric=current_rubric,
            rater_id=rater_id,
            dataset="representative"
        )
        results, current_kappa = run_essay_eval(config)

        stripped_results = [
            {k: v for k, v in result.items() if k != 'essay_text'} for result in results]

        rubric_history.append({
            "iteration": iteration,
            "rubric": current_rubric,
            "kappa": current_kappa,
            "results": stripped_results
        })

        print(f"Iteration {iteration}: Kappa = {current_kappa}")

        if current_kappa >= target_kappa:
            break

        current_rubric = improve_essay_rubric(
            rubric_history, rater_id, max_iterations)
        iteration += 1

    if current_kappa >= target_kappa:
        print(
            f"Target kappa of {target_kappa} reached. Running evaluation sample...")
        final_config = EssayEvalConfig(
            rubric=current_rubric,
            rater_id=rater_id,
            dataset="representative",
            num_examples=100
        )
        final_results, final_kappa = run_essay_eval(final_config)
        return current_rubric, final_results, final_kappa, rubric_history
    else:
        print(
            f"Failed to reach target kappa after {max_iterations} iterations.")
        return current_rubric, results, current_kappa, rubric_history


if __name__ == "__main__":
    initial_rubric = ["Is this a well-written essay for an 8th grade student?"]
    rater_id = 1
    target_kappa = 0.8

    optimized_rubric, final_results, final_kappa, rubric_history = optimize_rubric(
        initial_rubric, rater_id, target_kappa)

    print(f"Optimized Rubric: {json.dumps(optimized_rubric, indent=2)}")
    print(f"Final Kappa: {final_kappa}")
    print(f"Rubric History: {json.dumps(rubric_history, indent=2)}")
