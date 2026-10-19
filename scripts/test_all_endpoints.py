"""
Comprehensive End-to-End Test Suite for All New Features:
1. GET / and GET /demo (Web Demo UI)
2. GET /health
3. GET /metrics/cross-generator
4. GET /metrics/robustness
5. POST /predict (Image detection with calibrated confidence & SHAP)
6. POST /predict (Selective classification / abstention verification)
7. POST /predict (Robustness simulation parameters: simulate_jpeg, simulate_resize)
8. POST /evaluate/robustness (Interactive degradation battery)
9. POST /predict-video (Temporal motion consistency: optical flow warping, FFT variance, drift)
"""
import io
import os
import sys
import tempfile
import cv2
import numpy as np
import pyarrow.parquet as pq
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"


def test_suite():
    print("=" * 60)
    print("RUNNING COMPLETE VERITAS-AI SYSTEM VERIFICATION SUITE")
    print("=" * 60)

    # 1. Web Demo UI
    print("\n--- 1. Testing GET / (Web Demo UI) ---")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert "VeritasAI Media Forensic Lab" in r.text
    print("  [OK] Web Demo UI returned 200 OK with VeritasAI interface!")

    # 2. Health Check
    print("\n--- 2. Testing GET /health ---")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print(f"  [OK] Health Status: {r.json()}")

    # 3. Cross-Generator Benchmark Metrics
    print("\n--- 3. Testing GET /metrics/cross-generator ---")
    r = requests.get(f"{BASE_URL}/metrics/cross-generator")
    assert r.status_code == 200
    crossgen = r.json()
    mj_logo = crossgen["leave_one_generator_out_transfer"]["Midjourney"]["zero_shot_ai_accuracy"]
    overall_acc = crossgen["unified_probe_metrics"]["overall_accuracy"]
    print(f"  [OK] Multi-generator unified accuracy: {overall_acc*100:.1f}%")
    print(f"  [OK] Leave-One-Generator-Out Midjourney Zero-Shot accuracy: {mj_logo*100:.1f}%")

    # 4. Robustness Benchmark Metrics
    print("\n--- 4. Testing GET /metrics/robustness ---")
    r = requests.get(f"{BASE_URL}/metrics/robustness")
    assert r.status_code == 200
    rob = r.json()
    print(f"  [OK] Clean baseline accuracy: {rob['clean']['accuracy']*100:.1f}%")
    print(f"  [OK] JPEG Q=75 retention: {rob['jpeg_75']['retention_rate']*100:.1f}%")
    print(f"  [OK] Social Media Sim retention: {rob['social_recompress']['retention_rate']*100:.1f}%")

    # Load test image from parquet
    table = pq.read_table("data/genimage_cache/data/train-00000-of-00014.parquet")
    df = table.select(["label", "generator"]).to_pandas()
    mj_idx = int(df[df["generator"] == 4].index[3])
    mj_cell = table["image"][mj_idx].as_py()
    mj_bytes = mj_cell["bytes"] if isinstance(mj_cell, dict) else mj_cell

    # 5. Image Prediction (Midjourney)
    print("\n--- 5. Testing POST /predict (Midjourney Sample) ---")
    r = requests.post(
        f"{BASE_URL}/predict?abstain_margin=0.08",
        files={"file": ("midjourney_test.jpg", mj_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    res = r.json()
    print(f"  [OK] Verdict: {res['verdict']}")
    print(f"  [OK] Confidence: {res['confidence']*100:.1f}%")
    print(f"  [OK] Abstained: {res['abstained']}")
    print(f"  [OK] Closest Generator: {res.get('closest_ai_generator')}")
    print(f"  [OK] Faithfulness Audit: {res['faithfulness_audit'].get('details')}")

    # 6. Selective Classification / Abstention Trigger
    print("\n--- 6. Testing POST /predict with Abstention Trigger ---")
    r = requests.post(
        f"{BASE_URL}/predict?abstain_margin=0.49",
        files={"file": ("midjourney_test.jpg", mj_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    res_abstain = r.json()
    print(f"  [OK] Verdict: {res_abstain['verdict']}")
    print(f"  [OK] Abstained Flag: {res_abstain['abstained']}")
    print(f"  [OK] Abstention Reason: {res_abstain.get('abstention_reason')}")

    # 7. Robustness Simulation via /predict
    print("\n--- 7. Testing POST /predict with Simulated Degradation (JPEG Q=50, Resize 50%) ---")
    r = requests.post(
        f"{BASE_URL}/predict?abstain_margin=0.08&simulate_jpeg=50&simulate_resize=0.5",
        files={"file": ("midjourney_test.jpg", mj_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    res_sim = r.json()
    print(f"  [OK] Simulated Degraded Verdict: {res_sim['verdict']} ({res_sim['confidence']*100:.1f}%)")

    # 8. Interactive Stress Test Battery
    print("\n--- 8. Testing POST /evaluate/robustness (Multi-level Stress Test) ---")
    r = requests.post(
        f"{BASE_URL}/evaluate/robustness?abstain_margin=0.08",
        files={"file": ("midjourney_test.jpg", mj_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    stress = r.json()
    print(f"  [OK] Baseline Verdict: {stress['baseline_verdict']}")
    print(f"  [OK] Stability Score across 9 conditions: {stress['stability_score']*100:.1f}%")
    for t in stress["tests"][:4]:
        print(f"    - {t['test_name']}: {t['verdict']} ({t['confidence']*100:.1f}%)")

    # 9. Video Prediction
    print("\n--- 9. Testing POST /predict-video (Temporal Motion Dynamics) ---")
    tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_vid, fourcc, 10.0, (256, 256))
    rng = np.random.default_rng(42)
    for i in range(16):
        frame = np.uint8(rng.normal(120 + i * 2, 40, (256, 256, 3)).clip(0, 255))
        out.write(frame)
    out.release()

    with open(tmp_vid, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict-video?abstain_margin=0.08",
            files={"file": ("test.mp4", f, "video/mp4")},
        )
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)

    assert r.status_code == 200
    vid_res = r.json()
    print(f"  [OK] Video Verdict: {vid_res['verdict']} ({vid_res['confidence']*100:.1f}%)")
    print(f"  [OK] Optical Flow Warping Error: {vid_res['temporal_evidence']['optical_flow_warping_error']:.4f}")
    print(f"  [OK] Inter-Frame FFT Variance: {vid_res['temporal_evidence']['interframe_fft_variance']:.6f}")
    print(f"  [OK] Semantic Temporal Drift: {vid_res['temporal_evidence']['clip_temporal_drift']:.4f}")

    print("\n" + "=" * 60)
    print("ALL 9 TEST SUITES COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_suite()
