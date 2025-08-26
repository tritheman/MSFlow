#!/usr/bin/env bash
set -e
mkdir -p artifacts
# Chạy MLflow server với SQLite làm backend store và thư mục artifacts cục bộ
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5001