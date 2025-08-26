import argparse
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from utils import make_data
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

def build_model(model_type:str, params:dict):
    if model_type == "logreg":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params))
        ])
    elif model_type == "rf":
        clf = RandomForestClassifier(**params)
    else:
        raise ValueError("Unknown model_type. Use 'logreg' or 'rf'.")
    return clf

def plot_confusion_matrix(cm, out_png):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, int(z), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="cls-exp")
    parser.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--model-type", type=str, choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--params", type=str, default="{}")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=10)
    parser.add_argument("--class-sep", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Data
    X_train, X_test, y_train, y_test = make_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        class_sep=args.class_sep,
        random_state=args.random_state,
        test_size=args.test_size
    )

    # Params
    import ast
    params = ast.literal_eval(args.params)

    with mlflow.start_run(run_name=f"{args.model_type}") as run:
        mlflow.log_params({
            "model_type": args.model_type,
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "n_informative": args.n_informative,
            "class_sep": args.class_sep,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })
        mlflow.log_params({f"{args.model_type}_{k}": v for k, v in params.items()})

        # Build and fit
        model = build_model(args.model_type, params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        # Confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        os.makedirs("artifacts", exist_ok=True)
        cm_png = os.path.join("artifacts", "confusion_matrix.png")
        plot_confusion_matrix(cm, cm_png)
        mlflow.log_artifact(cm_png)

        # Signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2)
        )

        print("Run id:", run.info.run_id)
        print("Metrics:", {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

if __name__ == "__main__":
    main()