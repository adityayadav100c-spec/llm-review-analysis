# LLM Review Analysis Pipeline

Runs pretrained Hugging Face sentiment classifiers and a summarization model over a tiny, fictional car-dealership review set. It reports accuracy against the sample labels and wall-clock inference time. This is a reproducible demo, not a representative benchmark: the sample is small and intentionally mixed, so its accuracy should not be presented as a production result.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python analyze.py
```

First execution downloads model weights from Hugging Face and needs network access; CPU inference may be slow. Use the measured results only for the machine and model revisions on which you run the script.
