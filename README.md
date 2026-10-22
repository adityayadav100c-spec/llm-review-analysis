# LLM Review Analysis Pipeline

An end-to-end NLP evaluation demo for fictional dealership reviews. It compares
pretrained transformer sentiment models on the same labeled split, reports
class-level metrics and a bootstrap accuracy interval, exports row-level
errors, and produces an abstractive summary of a bounded review sample.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python analyze.py
```

For a larger deterministic synthetic dataset (600 examples by default):

```powershell
python generate_reviews.py --per-theme-and-sentiment 50
python analyze.py --max-reviews 600
```

First execution downloads model weights from Hugging Face and needs network
access; CPU inference can be slow. Timing separates model load from inference.
Scores only describe the selected demo sample and must not be presented as
production accuracy.

Run deterministic, dependency-light generator checks with
`python -m unittest discover -s tests`.

GitHub Actions validates dependencies, code compilation, and generator tests;
it does not download model weights or make API calls.

The checked-in generated sample has 600 rows across six synthetic journey
themes with balanced positive/negative labels. The report includes
theme-sliced error metrics. See [`docs/EVALUATION.md`](./docs/EVALUATION.md)
for the evaluation limits and data card.
