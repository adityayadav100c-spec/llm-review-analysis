import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import pipeline

ROOT = Path(__file__).parent
SENTIMENT_MODELS = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}
LABEL_MAPS = {
    "distilbert": {"positive": "positive", "negative": "negative"},
    "roberta": {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Compare sentiment models on labeled reviews.")
    parser.add_argument("--max-reviews", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.max_reviews < 1 or args.bootstrap_samples < 1:
        raise ValueError("max-reviews and bootstrap-samples must be positive.")
    generated = ROOT / "data" / "generated_reviews.csv"
    source = generated if generated.exists() else ROOT / "data" / "reviews.csv"
    reviews = pd.read_csv(source).head(args.max_reviews).copy()
    required_columns = {"review_id", "review", "sentiment"}
    missing_columns = required_columns - set(reviews.columns)
    if missing_columns:
        raise ValueError(f"Review dataset is missing required columns: {sorted(missing_columns)}")
    if reviews.empty or reviews["review"].isna().any():
        raise ValueError(f"Review data is empty or contains missing text: {source}")
    unexpected_sentiments = set(reviews["sentiment"].dropna()) - {"positive", "negative"}
    if reviews["sentiment"].isna().any() or unexpected_sentiments:
        raise ValueError(
            f"Expected positive/negative sentiment labels, got: "
            f"{sorted(unexpected_sentiments)}"
        )
    texts = reviews["review"].tolist()
    results = []
    prediction_columns = ["review_id", "review", "sentiment"]
    if "theme" in reviews:
        prediction_columns.append("theme")
    row_predictions = reviews[prediction_columns].copy()
    y_true = reviews["sentiment"].to_numpy()
    class_counts = reviews["sentiment"].value_counts()
    majority_label = sorted(class_counts.index)[
        int(np.argmax([class_counts[label] for label in sorted(class_counts.index)]))
    ]
    row_predictions["majority_baseline_prediction"] = majority_label
    row_predictions["majority_baseline_confidence"] = (
        class_counts[majority_label] / len(reviews)
    )
    bootstrap_rng = np.random.default_rng(42)
    bootstrap_indices = bootstrap_rng.integers(
        0, len(y_true), size=(args.bootstrap_samples, len(y_true))
    )
    majority_predictions = np.full(len(y_true), majority_label)
    majority_accuracy = accuracy_score(y_true, majority_predictions)
    majority_macro_f1 = f1_score(
        y_true,
        majority_predictions,
        labels=["negative", "positive"],
        average="macro",
        zero_division=0,
    )
    majority_bootstrap_accuracy = np.asarray(
        [
            accuracy_score(y_true[indices], majority_predictions[indices])
            for indices in bootstrap_indices
        ]
    )
    majority_bootstrap_f1 = np.asarray(
        [
            f1_score(
                y_true[indices],
                majority_predictions[indices],
                labels=["negative", "positive"],
                average="macro",
                zero_division=0,
            )
            for indices in bootstrap_indices
        ]
    )
    results.append(
        {
            "model": "majority_baseline",
            "model_id": None,
            "sample_count": len(texts),
            "accuracy": majority_accuracy,
            "accuracy_bootstrap_95ci_low": float(np.quantile(majority_bootstrap_accuracy, 0.025)),
            "accuracy_bootstrap_95ci_high": float(np.quantile(majority_bootstrap_accuracy, 0.975)),
            "macro_f1": majority_macro_f1,
            "macro_f1_bootstrap_95ci_low": float(np.quantile(majority_bootstrap_f1, 0.025)),
            "macro_f1_bootstrap_95ci_high": float(np.quantile(majority_bootstrap_f1, 0.975)),
            "model_load_seconds": 0.0,
            "inference_seconds": 0.0,
            "inference_seconds_per_review": 0.0,
        }
    )
    print(
        f"\nMajority baseline ({majority_label}) — "
        f"accuracy: {majority_accuracy:.3f}"
    )
    print(
        classification_report(
            y_true,
            majority_predictions,
            labels=["negative", "positive"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted; negative, positive):")
    print(confusion_matrix(y_true, majority_predictions, labels=["negative", "positive"]))
    prediction_columns_for_errors = ["majority_baseline_prediction"]
    for name, model_id in SENTIMENT_MODELS.items():
        load_start = time.perf_counter()
        classifier = pipeline("sentiment-analysis", model=model_id, device=-1)
        model_load_seconds = time.perf_counter() - load_start
        start = time.perf_counter()
        predictions = classifier(texts, truncation=True, batch_size=8)
        inference_seconds = time.perf_counter() - start
        labels = [
            LABEL_MAPS[name].get(item["label"].lower(), item["label"].lower())
            for item in predictions
        ]
        accuracy = accuracy_score(reviews["sentiment"], labels)
        bootstrap_accuracy = []
        y_pred = np.asarray(labels)
        bootstrap_macro_f1 = []
        for sample_indices in bootstrap_indices:
            bootstrap_accuracy.append(
                accuracy_score(y_true[sample_indices], y_pred[sample_indices])
            )
            bootstrap_macro_f1.append(
                f1_score(
                    y_true[sample_indices],
                    y_pred[sample_indices],
                    labels=["negative", "positive"],
                    average="macro",
                    zero_division=0,
                )
            )
        lower, upper = np.quantile(bootstrap_accuracy, [0.025, 0.975])
        f1_lower, f1_upper = np.quantile(bootstrap_macro_f1, [0.025, 0.975])
        row_predictions[f"{name}_prediction"] = labels
        row_predictions[f"{name}_confidence"] = [item["score"] for item in predictions]
        prediction_columns_for_errors.append(f"{name}_prediction")
        results.append(
            {
                "model": name,
                "model_id": model_id,
                "sample_count": len(texts),
                "accuracy": accuracy,
                "accuracy_bootstrap_95ci_low": float(lower),
                "accuracy_bootstrap_95ci_high": float(upper),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=["negative", "positive"],
                    average="macro",
                    zero_division=0,
                ),
                "macro_f1_bootstrap_95ci_low": float(f1_lower),
                "macro_f1_bootstrap_95ci_high": float(f1_upper),
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "inference_seconds_per_review": inference_seconds / len(texts),
            }
        )
        print(f"\n{name} — accuracy: {accuracy:.3f} ({lower:.3f}–{upper:.3f})")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=["negative", "positive"],
                zero_division=0,
            )
        )
        print("Confusion matrix (rows=true, columns=predicted; negative, positive):")
        print(confusion_matrix(y_true, y_pred, labels=["negative", "positive"]))

    benchmark = pd.DataFrame(results)
    print("\nRuntime comparison (load and inference measured separately):")
    print(
        benchmark[
            ["model", "model_load_seconds", "inference_seconds", "macro_f1"]
        ].to_string(index=False)
    )
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    benchmark.to_csv(output / "benchmark.csv", index=False)
    row_predictions["has_error"] = row_predictions.apply(
        lambda row: any(
            row[prediction_column] != row["sentiment"]
            for prediction_column in prediction_columns_for_errors
        ),
        axis=1,
    )
    row_predictions.to_csv(output / "row_predictions.csv", index=False)
    errors = row_predictions[row_predictions["has_error"]].copy()
    errors.head(25).to_csv(output / "error_review_sample.csv", index=False)
    theme_metrics = {}
    slice_model_names = ["majority_baseline", *SENTIMENT_MODELS]
    if "theme" in row_predictions:
        for theme, group in row_predictions.groupby("theme"):
            theme_metrics[str(theme)] = {
                name: {
                    "sample_count": int(len(group)),
                    "accuracy": float(
                        accuracy_score(group["sentiment"], group[f"{name}_prediction"])
                    ),
                    "macro_f1": float(
                        f1_score(
                            group["sentiment"],
                            group[f"{name}_prediction"],
                            labels=["negative", "positive"],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
                for name in slice_model_names
            }
    (output / "evaluation.json").write_text(
        json.dumps(
            {
                "source_file": source.name,
                "sample_count": len(reviews),
                "models": results,
                "journey_theme_slices": theme_metrics,
                "limitations": [
                    "Reviews are synthetic examples, not customer feedback.",
                    "Bootstrap intervals describe sampling variation within this demo only.",
                    "Model performance can change with domain, language, and model revisions.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summarizer = pipeline(
        "summarization", model="sshleifer/distilbart-cnn-12-6", device=-1
    )
    summary = summarizer(
        " ".join(texts[: min(20, len(texts))]),
        max_length=90,
        min_length=25,
        do_sample=False,
        truncation=True,
    )[0]["summary_text"]
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    import matplotlib.pyplot as plt

    plot_data = benchmark.set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    model_names = plot_data.index.tolist()
    accuracies = plot_data["accuracy"].to_numpy()
    lower_errors = accuracies - plot_data["accuracy_bootstrap_95ci_low"].to_numpy()
    upper_errors = plot_data["accuracy_bootstrap_95ci_high"].to_numpy() - accuracies
    axes[0].bar(
        model_names,
        accuracies,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=4,
        color=["#8c8c8c" if name == "majority_baseline" else "#3274a1" for name in model_names],
    )
    axes[0].set_title("Accuracy (95% bootstrap interval)")
    axes[0].set_ylim(0, 1)
    axes[1].bar(
        model_names,
        plot_data["macro_f1"],
        color=["#8c8c8c" if name == "majority_baseline" else "#e1812c" for name in model_names],
    )
    axes[1].set_title("Macro F1")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "model_comparison.png", dpi=160)
    plt.close(figure)
    if theme_metrics:
        figure, axis = plt.subplots(figsize=(10, 5))
        themes = list(theme_metrics)
        positions = np.arange(len(themes))
        bar_width = 0.8 / len(slice_model_names)
        for index, name in enumerate(slice_model_names):
            scores = [theme_metrics[theme][name]["macro_f1"] for theme in themes]
            axis.bar(
                positions + index * bar_width,
                scores,
                bar_width,
                label=name.replace("_", " "),
            )
        axis.set_xticks(positions + bar_width * (len(slice_model_names) - 1) / 2)
        axis.set_xticklabels(themes, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Macro F1")
        axis.set_title("Sentiment performance by synthetic journey theme")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "theme_performance.png", dpi=160)
        plt.close(figure)

    report_lines = [
        "# Review Sentiment Model Comparison",
        "",
        f"- Source: `{source.name}`",
        f"- Scored rows: {len(reviews)}",
        f"- Majority baseline label: `{majority_label}`",
        "",
        "| Model | Accuracy (95% bootstrap interval) | Macro F1 | Inference seconds |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        report_lines.append(
            f"| {result['model']} | {result['accuracy']:.3f} "
            f"({result['accuracy_bootstrap_95ci_low']:.3f}–"
            f"{result['accuracy_bootstrap_95ci_high']:.3f}) | "
            f"{result['macro_f1']:.3f} | {result['inference_seconds']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "These examples are synthetic and intentionally patterned. Metrics and "
            "bootstrap intervals describe this generated sample only; they are not "
            "estimates of performance on real dealership customers. Review "
            "`error_review_sample.csv` and evaluate on an independently labeled, "
            "representative dataset before choosing a deployment model.",
            "",
            f"Summary model output: {summary}",
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"Saved benchmark, metrics, errors, charts, and report under {output}")


if __name__ == "__main__":
    main()
