import argparse
import os
import pandas as pd
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="cls-exp")
    parser.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--metric", type=str, default="f1")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    keep = ["run_id", "params.model_type", "metrics.accuracy", "metrics.precision", "metrics.recall", "metrics.f1"]
    df = df[keep].sort_values(by=f"metrics.{args.metric}", ascending=False)
    print(df.to_string(index=False))
    df.to_csv("results.csv", index=False)
    print("Saved results.csv")

if __name__ == "__main__":
    main()