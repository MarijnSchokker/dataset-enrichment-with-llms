import mlflow
import typer

import plotly.express as px

app = typer.Typer()


@app.command()
def read_and_summarize_mlflow_runs(read_from_experiment: str, write_to_experiment: str):
    # read all runs from MLFlow experiment `read_from_experiment`

    mlflow.set_experiment(write_to_experiment)
    with mlflow.start_run():
        experiment = mlflow.get_experiment_by_name(read_from_experiment)
        experiment_id = experiment.experiment_id
        runs_df = mlflow.search_runs(experiment_ids=experiment_id)

        # percentage correct per method
        fig = px.strip(
            runs_df,
            x="params.temperature",
            y="metrics.percentage_correct",
            color="params.method",
        )
        mlflow.log_figure(fig, "points_plot.html")

        # bar chart with #correct responses
        fig = px.bar(
            runs_df,
            x="params.temperature",
            y="metrics.number_correct",
            color="params.method",
            barmode="group",
        )
        mlflow.log_figure(fig, "bar_no_of_correct_responses.html")

        # bar chart with mean percentage correct
        fig = px.bar(
            runs_df.groupby(["params.method", "params.temperature"]).mean().reset_index(),
            x="params.temperature",
            y="metrics.percentage_correct",
            color="params.method",
            barmode="group",
        )
        mlflow.log_figure(fig, "bar.html")

        # violin chart with mean percentage correct
        fig = px.violin(
            runs_df.sort_values(by="params.method"),
            x="params.temperature",
            y="metrics.percentage_correct",
            color="params.method",
            points="all",
        )
        mlflow.log_figure(fig, "violin.html")

        # failures per method
        did_run_fail = runs_df["status"] == "FAILED"
        failed_runs = runs_df[did_run_fail]
        method_failures = failed_runs.groupby("params.method").count()[["run_id"]]

        # total queries
        finished_runs = runs_df[
            (runs_df["status"] == "FINISHED")
            &
            (runs_df["metrics.percentage_correct"] >= 0)
        ]
        finishes_per_method = finished_runs.groupby(["params.temperature", "params.method"]).count()[["run_id"]]

        # log overview
        mlflow.log_text(
            f"""
            <h1># of finished runs per method</h1>
            {finishes_per_method.to_html()}
            <h1>Failures per method</h1>
            {method_failures.to_html()}
            """,
            "overview.html"
        )
