import argparse
import json
from retail_pulse_pipeline import train_and_save


def main():
    parser = argparse.ArgumentParser(description="Train the Retail Pulse product category model.")
    parser.add_argument("--data", default="retail_sales_dataset.csv", help="Path to the retail sales CSV file.")
    parser.add_argument("--output", default="retail_pulse_model.joblib", help="Output path for the trained model.")
    parser.add_argument("--metrics", default="training_metrics.json", help="Output path for the training metrics JSON file.")
    args = parser.parse_args()

    model, metrics = train_and_save(data_path=args.data, model_path=args.output)

    with open(args.metrics, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Training complete. Model saved to {args.output}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
