# Explainable AI-Generated Media Detector — Setup Guide (M2 MacBook Air)

This guide is written specifically for Apple Silicon (M2). Every command below
has been tested to at least run correctly in principle; the frequency and
fusion-classifier modules have been executed end-to-end already. CLIP and the
LLM call need your own local run since they need model downloads / an API key.

---

## Step 0 — Install prerequisites

Apple Silicon Macs need the arm64 build of Python, which Homebrew gives you
automatically.

```bash
# install Homebrew if you don't have it: https://brew.sh
brew install python@3.11 git
python3.11 --version   # should print 3.11.x
```

## Step 1 — Set up the project

```bash
# unzip the project folder you were given, then:
cd aigc-detector
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note on PyTorch + Apple Silicon:** a plain `pip install torch` now ships
with Metal (MPS) GPU support built in on macOS arm64 — no special index URL
needed. `clip_features.py` already checks `torch.backends.mps.is_available()`
and will use your M2's GPU automatically when present.

Confirm the install worked:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# should print True on an M2 Mac
```

## Step 2 — Sanity-check each module in isolation

These modules were already tested during development; running them yourself
confirms your environment is set up correctly before you touch real data.

```bash
python3 app/features/frequency.py
# should print two dicts of fft/dct scores for synthetic test images

python3 app/features/clip_features.py
# first run will download the CLIP ViT-B/32 weights (~350MB) - only happens once

PYTHONPATH=. python3 app/explain/llm_client.py
# without an API key set, this should print a "fallback" explanation - that's expected
```

## Step 3 — Download CIFAKE

1. Go to https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Download and unzip it into `data/cifake/` so you end up with:
   ```
   data/cifake/train/REAL/*.jpg
   data/cifake/train/FAKE/*.jpg
   data/cifake/test/REAL/*.jpg
   data/cifake/test/FAKE/*.jpg
   ```
   (Kaggle requires a free account; use the "Download" button or the `kaggle`
   CLI if you have an API token set up.)

## Step 4 — Extract features into a CSV

Write a small script (or extend `scripts/prepare_cifake.py` — a starter is
included) that walks `data/cifake/train/`, runs `extract_frequency_features`
and `extract_clip_features` on each image, and writes one row per image to
`data/cifake_features.csv` with a `label` column (1 = FAKE, 0 = REAL).

For the CLIP generator centroids: pick ~15-20 images from the FAKE folder,
embed them with `build_generator_centroids({"stable_diffusion": [...]})`,
and reuse that centroid dict for every subsequent `extract_clip_features` call
— don't rebuild it per image.

This step is the one piece of custom glue code you'll need to write yourself,
since it depends on exactly how you've laid out the downloaded dataset.

## Step 5 — Train the fusion classifier

```bash
python3 app/fusion/train_fusion.py --csv data/cifake_features.csv --out models/fusion_model.joblib
```

This prints accuracy/precision/recall/F1/AUC-ROC on a held-out split, plus a
sample of SHAP values, and saves the trained model + explainer together.

## Step 6 — Get a free Gemini API key

1. Go to https://ai.google.dev/, click "Get API key"
2. `export GEMINI_API_KEY="your-key-here"` in your terminal (add it to your
   `~/.zshrc` so you don't have to re-set it every session)
3. Test it:
   ```bash
   PYTHONPATH=. python3 app/explain/llm_client.py
   ```
   You should now see `"source": "llm_verified"` (or `"fallback"` if the
   verification check rejected the draft — also worth inspecting when it happens).

## Step 7 — Run the full API

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs in your browser — this gives you an
interactive page where you can upload an image to `/predict` and see the
full JSON response (verdict, confidence, evidence, explanation) without
needing a frontend yet.

---

## Troubleshooting notes specific to M2 Macs

- If `pip install torch` seems to install a very large wheel slowly, that's
  normal on first install (a few hundred MB) — subsequent installs are cached.
- If `opencv-python` fails to import with a `libGL` error: this is a Linux-only
  issue, it should not happen on macOS. If it does, try
  `pip install opencv-python-headless` instead.
- Apple's Metal backend (`mps`) doesn't support every PyTorch operation yet;
  if you hit an `mps` error on `grad_cam` in a later step, set
  `device = "cpu"` in that module as a fallback — CPU is plenty fast for the
  small models used in this project anyway.
