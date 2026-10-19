"""
Builds the constrained prompt sent to the free-tier LLM. The model only ever
sees this text - never the image - which is what makes the resulting
explanation checkable against the evidence dict.
"""

SYSTEM_PROMPT = """You are an evidence-reporting assistant for an AI-generated media detector.
You will be given a structured evidence record produced by a classifier. You did not see
the original image or video - you only have the evidence record below.

Your task: write a short, factual explanation (2-3 sentences) of why the system reached
its verdict, using ONLY the fields provided.

STRICT RULES:
1. Do not mention any visual detail, object, color, or scene element that is not a field
   in the evidence record. You have not seen the image.
2. Do not use words like "clearly", "obviously", or "definitely" - describe the evidence
   plainly, without embellishment.
3. Every claim you make must be traceable to a specific field and value below.
4. If a field is not present or its value is near zero/neutral, do not mention it.
5. If verdict is 'uncertain' or 'abstained', state that the model withheld judgment due to borderline signals.
6. Do not speculate about the generator, artist, or subject beyond what evidence explicitly states.
7. Output ONLY the explanation text. No preamble, no headers, no markdown."""

FEWSHOT_USER = """Evidence record:
- verdict: AI-generated
- confidence: 0.97
- clip_semantic_score: 0.92
- fft_anomaly_score: 0.45
- top_evidence: ["clip_semantic_score", "fft_anomaly_score"]

Write the explanation now, following all rules above."""

FEWSHOT_ASSISTANT = """The image was classified as AI-generated with 97% confidence.
The strongest signal was a high semantic anomaly score (0.92) indicating alignment with synthetic generator embeddings. Frequency-domain analysis also exhibited characteristic high-frequency artifacts (0.45), reinforcing the verdict."""

FEWSHOT_ABSTAIN_USER = """Evidence record:
- verdict: uncertain
- confidence: 0.85
- abstained: True
- calibrated_prob_ai: 0.485
- clip_semantic_score: 0.49
- fft_anomaly_score: 0.42
- top_evidence: ["clip_semantic_score", "fft_anomaly_score"]

Write the explanation now, following all rules above."""

FEWSHOT_ABSTAIN_ASSISTANT = """The detector abstained from a definitive verdict, classifying the media as uncertain. The calibrated probability (0.485) falls within the decision margin as both semantic consistency (0.49) and frequency features (0.42) are borderline. Manual forensic review is recommended."""

FEWSHOT_VIDEO_USER = """Evidence record:
- media_type: video
- verdict: AI-generated
- confidence: 0.93
- optical_flow_warping_error: 0.28
- interframe_fft_variance: 0.0042
- clip_temporal_drift: 0.081
- top_evidence: ["optical_flow_warping_error", "clip_temporal_drift"]

Write the explanation now, following all rules above."""

FEWSHOT_VIDEO_ASSISTANT = """The video was classified as AI-generated with 93% confidence.
The primary indicators were a high inter-frame optical flow warping error (0.28) and notable semantic temporal drift (0.081), pointing to unnatural frame-to-frame pixel morphing characteristic of synthetic video models."""


def format_evidence_for_prompt(evidence: dict) -> str:
    """Turns the evidence dict from classifier.py into readable prompt text."""
    lines = [f"- verdict: {evidence['verdict']}", f"- confidence: {evidence['confidence']}"]
    if "media_type" in evidence:
        lines.append(f"- media_type: {evidence['media_type']}")
    if evidence.get("abstained"):
        lines.append("- abstained: True")
    if "calibrated_prob_ai" in evidence:
        lines.append(f"- calibrated_prob_ai: {evidence['calibrated_prob_ai']}")
    for key, value in evidence.items():
        if key in ("verdict", "confidence", "top_evidence", "media_type", "abstained", "calibrated_prob_ai", "raw_prob_ai", "abstain_margin", "abstention_reason"):
            continue
        if isinstance(value, dict) and "value" in value:
            lines.append(f"- {key}: {value['value']}")
        else:
            lines.append(f"- {key}: {value}")
    if "top_evidence" in evidence:
        lines.append(f"- top_evidence: {evidence['top_evidence']}")
    return "\n".join(lines)


def build_messages(evidence: dict) -> list:
    is_video = evidence.get("media_type") == "video"
    is_abstain = evidence.get("abstained", False) or evidence.get("verdict") == "uncertain"
    user_prompt = f"Evidence record:\n{format_evidence_for_prompt(evidence)}\n\nWrite the explanation now, following all rules above."

    if is_abstain:
        fewshot_u = FEWSHOT_ABSTAIN_USER
        fewshot_a = FEWSHOT_ABSTAIN_ASSISTANT
    elif is_video:
        fewshot_u = FEWSHOT_VIDEO_USER
        fewshot_a = FEWSHOT_VIDEO_ASSISTANT
    else:
        fewshot_u = FEWSHOT_USER
        fewshot_a = FEWSHOT_ASSISTANT

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": fewshot_u},
        {"role": "assistant", "content": fewshot_a},
        {"role": "user", "content": user_prompt},
    ]


if __name__ == "__main__":
    example_evidence = {
        "verdict": "uncertain",
        "confidence": 0.82,
        "abstained": True,
        "calibrated_prob_ai": 0.49,
        "clip_semantic_score": {"value": 0.51, "contribution": 0.12},
        "fft_anomaly_score": {"value": 0.40, "contribution": -0.05},
        "top_evidence": ["clip_semantic_score", "fft_anomaly_score"],
    }
    for m in build_messages(example_evidence):
        print(f"[{m['role']}]\n{m['content']}\n")
