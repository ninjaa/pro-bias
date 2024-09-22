import json
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from jinja2 import Template
from src.utils.run_essay_eval import run_essay_eval, EssayEvalConfig
from src.prompts.improve_essay_rubric import improve_essay_rubric


def save_to_json(data, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)

    full_filename = backup_dir / f"{filename}_{timestamp}.json"

    with open(full_filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Backup saved to {full_filename}")


def create_html_report(backup_data):
    # Create kappa history graph
    kappa_history = [item['kappa'] for item in backup_data['rubric_history']]
    iterations = list(range(len(kappa_history)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=kappa_history,
                  mode='lines+markers', name='Kappa'))
    fig.add_hline(y=backup_data['target_kappa'],
                  line_dash="dash", annotation_text="Target Kappa")
    fig.update_layout(title='Kappa Progress Over Iterations',
                      xaxis_title='Iteration', yaxis_title='Kappa')

    kappa_graph = fig.to_html(full_html=False)

    # Prepare HTML template
    template = Template('''
    <html>
    <head>
        <title>Rubric Optimization Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            h1, h2 { color: #333; }
            .container { max-width: 800px; margin: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rubric Optimization Report</h1>
            <p>Initial Rubric: {{ initial_rubric }}</p>
            <p>Target Kappa: {{ target_kappa }}</p>
            <h2>Optimization Process</h2>
            {{ kappa_graph | safe }}
            <h2>Final Results</h2>
            <p>Best Rubric: {{ best_rubric }}</p>
            <p>Best Kappa (Representative Set): {{ best_kappa }}</p>
            <p>Final Kappa (Evaluation Set): {{ final_kappa }}</p>
        </div>
    </body>
    </html>
    ''')

    # Render HTML
    html_content = template.render(
        initial_rubric=backup_data['initial_rubric'],
        target_kappa=backup_data['target_kappa'],
        kappa_graph=kappa_graph,
        best_rubric=backup_data['best_rubric'],
        best_kappa=backup_data['best_kappa'],
        final_kappa=backup_data['final_kappa']
    )

    # Save HTML report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / \
        f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report saved to {report_path}")


def optimize_rubric(initial_rubric, rater_id, target_kappa, max_iterations=10):
    rubric_history = []
    iteration = 0
    current_rubric = initial_rubric
    best_rubric = initial_rubric
    best_kappa = 0

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
            "kappa_delta": target_kappa - current_kappa,
            "results": stripped_results
        })

        print(f"Iteration {iteration}: Kappa = {current_kappa}")

        # Update best_rubric if current_kappa is better
        if current_kappa > best_kappa:
            best_rubric = current_rubric
            best_kappa = current_kappa

        if current_kappa >= target_kappa:
            break

        current_rubric = improve_essay_rubric(
            rubric_history, rater_id, max_iterations, target_kappa)
        iteration += 1

    print(f"Best kappa achieved: {best_kappa}")
    print(f"Running final evaluation with best rubric... {best_rubric}")
    final_config = EssayEvalConfig(
        rubric=best_rubric,
        rater_id=rater_id,
        dataset="evaluation",
        num_examples=10
    )
    final_results, final_kappa = run_essay_eval(final_config)

    # Prepare data for JSON backup
    backup_data = {
        "initial_rubric": initial_rubric,
        "rater_id": rater_id,
        "target_kappa": target_kappa,
        "max_iterations": max_iterations,
        "best_rubric": best_rubric,
        "best_kappa": best_kappa,
        "final_kappa": final_kappa,
        "rubric_history": rubric_history,
        "final_results": final_results
    }

    # Save backup
    save_to_json(backup_data, "rubric_optimization")

    # Create HTML report
    create_html_report(backup_data)

    return best_rubric, final_results, final_kappa, rubric_history


if __name__ == "__main__":
    initial_rubric = ["Is this a well-written essay for an 8th grade student?"]
    rater_id = 1
    target_kappa = 0.7

    optimized_rubric, final_results, final_kappa, rubric_history = optimize_rubric(
        initial_rubric, rater_id, target_kappa)

    print(f"Optimized Rubric: {json.dumps(optimized_rubric, indent=2)}")
    print(f"Final Kappa: {final_kappa}")
    print(f"Rubric History: {json.dumps(rubric_history, indent=2)}")
