import argparse
import itertools
import os
import time
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from train_one import main as train_once

def run_grid(args):
    # Build grids
    grids = []

    # Logistic Regression grid
    logreg_grid = {
        "model_type": ["logreg"],
        "params": [
            {"C": 0.1, "max_iter": 200, "solver": "lbfgs"},
            {"C": 1.0, "max_iter": 300, "solver": "lbfgs"},
            {"C": 10.0, "max_iter": 400, "solver": "lbfgs"}
        ]
    }

    # Random Forest grid
    rf_grid = {
        "model_type": ["rf"],
        "params": [
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "n_jobs": -1},
            {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2, "n_jobs": -1},
            {"n_estimators": 300, "max_depth": 5, "min_samples_split": 4, "n_jobs": -1}
        ]
    }

    for grid in [logreg_grid, rf_grid]:
        for mtype in grid["model_type"]:
            for p in grid["params"]:
                yield mtype, p

def call_train_cli(args, model_type, params):
    # Build fake argv for train_one.main
    import sys
    import shlex
    params_str = str(params)
    argv = [
        "train_one.py",
        "--experiment-name", args.experiment_name,
        "--tracking-uri", args.tracking_uri,
        "--model-type", model_type,
        "--params", params_str,
        "--n-samples", str(args.n_samples),
        "--n-features", str(args.n_features),
        "--n-informative", str(args.n_informative),
        "--class-sep", str(args.class_sep),
        "--test-size", str(args.test_size),
        "--random-state", str(args.random_state),
    ]
    # Save and replace sys.argv temporarily
    old_argv = sys.argv
    sys.argv = argv
    try:
        from train_one import main as train_main
        train_main()
    finally:
        sys.argv = old_argv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="cls-exp")
    parser.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "accuracy", "precision", "recall"])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=10)
    parser.add_argument("--class-sep", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Run grid
    for model_type, params in run_grid(args):
        print(f"Running {model_type} with {params}")
        call_train_cli(args, model_type, params)

    # Select best run
    client = MlflowClient(tracking_uri=args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=[f"metrics.{args.metric} DESC"])
    if df.empty:
        print("No runs found")
        return
    best = df.iloc[0]
    best_run_id = best["run_id"]
    best_metric = best[f"metrics.{args.metric}"]
    print(f"Best run id: {best_run_id} metric {args.metric} = {best_metric}")

    # Register best model
    model_name = "BestClassifier"
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Wait until ready then transition to Production
    for _ in range(20):
        mv = client.get_model_version(name=model_name, version=result.version)
        status = mv.status
        if status == "READY":
            break
        time.sleep(1)

    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Registered {model_name} v{result.version} to Production")

if __name__ == "__main__":
    main()