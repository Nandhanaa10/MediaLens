# Explainable AI-Generated Media Detector (Image & Video)

An end-to-end, interpretable forensic detection system capable of identifying AI-generated images and videos across modern generative models (including **Midjourney, Stable Diffusion 1.5, GLIDE, BigGAN, ADM, VQDM, and Wukong**) alongside real physical media.

Unlike black-box neural detectors that suffer from catastrophic shortcut learning, this system integrates:
1. **Spatial Frequency Forensics:** 2D Fast Fourier Transform (FFT) radial anomaly scoring and Discrete Cosine Transform (DCT) high-frequency block residual extraction.
2. **Semantic Representation & Contrastive Fingerprinting:** Universal linear probing on frozen CLIP ViT-B/32 representations (Ojha et al., CVPR 2023) and multi-generator contrastive centroid margins ($\Delta_{\text{sim}} = \text{sim}_{\text{AI}} - \text{sim}_{\text{Real}}$).
3. **Temporal Motion Consistency (Video):** Dense Farnebäck optical flow warping discrepancy, inter-frame FFT variance (temporal jitter/flicker), and CLIP inter-frame semantic drift.
4. **Interpretable Fusion with Exact SHAP Attribution:** Scikit-Learn fusion classifier with exact Shapley value attributions for every forensic signal.
5. **Grounded Multi-Modal Explanation:** Free-tier Gemini 3.8 Flash generation with strict factual constraints and deterministic numerical verification.

---

## Performance Summary

Evaluated on the **GenImage** high-resolution benchmark (held-out test split):
- **Overall AUC-ROC:** **`92.7%`** (0.9269)
- **Midjourney Detection Accuracy:** **`96.6%`** (Mean $P(\text{AI}) = 0.879$)
- **Real Photo Specificity:** **`83.5%`** (Mean $P(\text{AI}) = 0.206$)
- **BigGAN Accuracy:** **`96.4%`** | **GLIDE Accuracy:** **`93.1%`** | **Wukong Accuracy:** **`92.9%`**

---

## System Requirements

- **Python:** 3.10 or 3.11 (Recommended)
- **OS:** Windows 10/11 or macOS (Intel / Apple Silicon M1/M2/M3/M4)
- **Hardware:** Runs efficiently on CPU (no dedicated GPU required, optional Apple Silicon Metal `mps` support built-in)
- **API Key (Optional for LLM explanations):** Free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

---

## Setup & Installation

### Option A: Windows (PowerShell)

```powershell
# 1. Clone or navigate to the repository
cd C:\path\to\aigc-detector

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1
# If PowerShell execution policy restricts scripts, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: macOS (Terminal / bash / zsh)

```bash
# 1. Clone or navigate to the repository
cd /path/to/aigc-detector

# 2. Ensure Homebrew Python 3.11 is installed (on Apple Silicon arm64)
# brew install python@3.11 git

# 3. Create a virtual environment
python3 -m venv venv

# 4. Activate the virtual environment
source venv/bin/activate

# 5. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

*Note on PyTorch on Apple Silicon:* `torch` automatically enables Metal Performance Shaders (`mps`) GPU acceleration out of the box on macOS arm64.

---

## Environment Configuration

To enable grounded natural-language explanations via Gemini 3.8 Flash:

### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

### macOS / Linux:
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

*(If no key is configured, the detector automatically falls back to deterministic template explanations without throwing errors.)*

---

## Running the Application

Start the FastAPI application with Uvicorn:

### Windows:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### macOS:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started, open your web browser to:
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Health & Model Version:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Usage & Endpoints

### 1. Single Image Detection (`POST /predict`)
Uploads an image file (`.jpg`, `.png`, `.webp`) and evaluates spatial frequency anomalies, semantic generator fingerprints, and SHAP attributions.

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@sample_midjourney.jpg"
```

**Example Response:**
```json
{
  "verdict": "AI-generated",
  "confidence": 0.976,
  "closest_ai_generator": "midjourney",
  "model_version": "v3_genimage",
  "evidence": {
    "verdict": "AI-generated",
    "confidence": 0.976,
    "clip_semantic_score": { "value": 0.7616, "contribution": 3.5714 },
    "clip_contrast_margin": { "value": 0.0500, "contribution": 0.9146 },
    "clip_max_generator_similarity": { "value": 0.7941, "contribution": -0.2321 },
    "fft_anomaly_score": { "value": 0.4033, "contribution": -0.0021 },
    "dct_anomaly_score": { "value": 0.0835, "contribution": -0.0369 },
    "top_evidence": ["clip_semantic_score", "clip_contrast_margin"]
  },
  "explanation": "The image was classified as AI-generated with 98% confidence. The primary indicator was a high semantic anomaly score (0.7616), reinforced by strong alignment with Midjourney generator fingerprints (0.05 margin).",
  "explanation_source": "llm_verified"
}
```

---

### 2. Video Temporal Consistency Detection (`POST /predict-video`)
Uploads a video (`.mp4`, `.mov`, `.avi`, `.webm`) and extracts uniform frames to detect temporal motion inconsistencies, optical flow warping error, and inter-frame frequency variance.

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict-video" \
     -F "file=@synthetic_video.mp4"
```

**Example Response:**
```json
{
  "media_type": "video",
  "verdict": "AI-generated",
  "confidence": 0.946,
  "metadata": {
    "fps": 24.0,
    "total_frames": 120,
    "sampled_frames": 16,
    "duration_seconds": 5.0
  },
  "temporal_evidence": {
    "optical_flow_warping_error": 0.0844,
    "interframe_fft_variance": 0.000094,
    "clip_temporal_drift": 0.0242
  },
  "evidence": {
    "verdict": "AI-generated",
    "confidence": 0.946,
    "top_evidence": ["clip_semantic_score", "optical_flow_warping_error"]
  },
  "explanation": "The video was classified as AI-generated with 95% confidence due to unnatural inter-frame optical flow warping error (0.0844) and persistent synthetic semantic artifacts across sampled frames."
}
```

---

## Dataset Preparation & Re-training

To reproduce the multi-generator benchmark or re-train on custom data:

1. **Extract and Train on GenImage (2,000 samples):**
   ```bash
   # Windows
   python scripts/prepare_genimage.py

   # macOS
   python3 scripts/prepare_genimage.py
   ```
   This automatically downloads or reads `data/genimage_cache/`, extracts 512-dim CLIP representations and 2D frequency features, fits the universal linear probe, computes unit-normalized contrastive centroids, and exports:
   - `models/genimage_centroids.joblib`
   - `models/genimage_fusion_model.joblib`
   - `data/genimage_features.csv`

2. **Verify Module Health Individually:**
   ```bash
   # Test spatial frequency analysis
   python app/features/frequency.py

   # Test video temporal consistency
   python app/features/temporal.py

   # Test LLM prompt builder and verification
   python app/explain/llm_client.py
   ```

---

## Project Structure

```
aigc-detector/
│
├── app/
│   ├── features/
│   │   ├── frequency.py         # 2D FFT & DCT frequency anomaly scores
│   │   ├── clip_features.py     # Universal CLIP probe & contrastive similarity margins
│   │   └── temporal.py          # Farnebäck optical flow warping & inter-frame FFT variance
│   ├── fusion/
│   │   ├── classifier.py        # Inference wrapper with exact SHAP Shapley values
│   │   └── train_fusion.py      # Logistic Regression training pipeline with StandardScaler
│   ├── explain/
│   │   ├── prompt_template.py   # Strictly constrained few-shot evidence prompts
│   │   └── llm_client.py        # Gemini 3.8 Flash caller with numeric verification fallback
│   ├── video/
│   │   └── frame_extractor.py   # Uniform sampling and resolution standardizer
│   └── main.py                  # FastAPI backend with /predict and /predict-video
│
├── data/                        # Datasets and feature tables (genimage_features.csv)
├── models/                      # Trained fusion models and generator centroid dictionaries
├── scripts/                     # Dataset processing pipelines (prepare_genimage.py)
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## License

This project is licensed under the MIT License.
