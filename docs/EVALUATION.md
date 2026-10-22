# Evaluation Protocol and Data Card

## Dataset

`data/generated_reviews.csv` contains 600 deterministic synthetic reviews:
six dealership journey themes crossed with positive and negative sentiment.
The theme/sentiment combinations are balanced by construction and each row is
marked `is_synthetic=true`. `generate_reviews.py` recreates the sample from a
seed. These are controlled examples, not a representative customer survey.

## Measurements

- Accuracy and macro F1, using the same review rows for every pretrained model.
- A per-class precision/recall/F1 report and confusion matrix.
- A 95% percentile bootstrap interval for accuracy, resampling review rows
  with a fixed random seed.
- A majority-class baseline using the same labels and rows as the transformer
  classifiers.
- Model loading time and inference-only time, reported separately on CPU.
- Row-level predictions and error flags; theme-slice metrics when theme labels
  are available, plus a small reviewable error sample.
- A pretrained summarizer applied to at most the first 20 sample reviews to
  bound input size.

No model is trained or tuned on this sample. The bootstrap interval describes
resampling variation within these rows; it does not represent uncertainty over
all dealership customers or new model versions. Because generated examples
share controlled writing patterns with their labels, scores are not production
performance. Model comparison should be repeated with a held-out,
representative, human-labeled set before selecting a deployment model.

The model IDs are recorded in the generated report. Inference speed depends on
hardware, software versions, batch size, and model download/cache state.
