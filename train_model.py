import argparse
import json
from pathlib import Path

from ml_pipeline import train_and_save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CyberVision model")
    parser.add_argument("--dataset", default="data/dataset_alt", help="Dataset root with virus/ and non_virus/")
    parser.add_argument("--model-out", default="models/cybervision_model.joblib", help="Output model path")
    parser.add_argument("--metrics-out", default="models/train_metrics.json", help="Output metrics path")
    args = parser.parse_args()

    metrics = train_and_save_model(args.dataset, args.model_out)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 68)
    print("CYBERVISION - TRAINING DONE")
    print("=" * 68)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall   : {metrics['recall_weighted']:.4f}")
    print(f"F1-score : {metrics['f1_weighted']:.4f}")
    print("-" * 68)
    print(metrics["classification_report"])
    print("-" * 68)
    print(f"Model   : {args.model_out}")
    print(f"Metrics : {args.metrics_out}")


if __name__ == "__main__":
    main()
