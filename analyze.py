import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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
    reviews = pd.read_csv(ROOT / "data" / "reviews.csv")
    texts = reviews["review"].tolist()
    results = []
    for name, model_id in SENTIMENT_MODELS.items():
        classifier = pipeline("sentiment-analysis", model=model_id, device=-1)
        start = time.perf_counter()
        predictions = classifier(texts, truncation=True, batch_size=8)
        elapsed = time.perf_counter() - start
        labels = [
            LABEL_MAPS[name].get(item["label"].lower(), item["label"].lower())
            for item in predictions
        ]
        accuracy = accuracy_score(reviews["sentiment"], labels)
        results.append(
            {
                "model": name,
                "model_id": model_id,
                "accuracy_on_tiny_demo": accuracy,
                "seconds_for_sample": elapsed,
                "seconds_per_review": elapsed / len(texts),
            }
        )
        print(f"\n{name} — accuracy on tiny sample: {accuracy:.3f}")
        print(classification_report(reviews["sentiment"], labels, zero_division=0))

    benchmark = pd.DataFrame(results)
    print("\nTiming (includes model inference only; model loading excluded):")
    print(benchmark[["model", "seconds_for_sample", "seconds_per_review"]].to_string(index=False))
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    benchmark.to_csv(output / "benchmark.csv", index=False)
    reviews = reviews.copy()
    summarizer = pipeline(
        "summarization", model="sshleifer/distilbart-cnn-12-6", device=-1
    )
    summary = summarizer(
        " ".join(texts), max_length=90, min_length=25, do_sample=False
    )[0]["summary_text"]
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"Saved benchmark and summary under {output}")


if __name__ == "__main__":
    main()
