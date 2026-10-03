"""
Calls a free-tier LLM (Gemini's free tier is the easiest to get working
quickly - https://ai.google.dev/, click "Get API key"), and runs the
faithfulness verification step from the design doc: reject and fall back
to a deterministic template if the draft mentions anything not traceable
to the evidence dict.

Set your API key as an environment variable before running:
    export GEMINI_API_KEY="your-key-here"
"""
import os
import re
import sys
from pathlib import Path
import requests

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.explain.prompt_template import build_messages

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash"
]


def call_gemini(messages: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY as an environment variable first.")

    flattened = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)

    last_error = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        gen_config = {
            "temperature": 0.2,
            "maxOutputTokens": 300,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        }

        try:
            print(f"Requesting explanation from {model}...")
            response = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": flattened}]}],
                    "generationConfig": gen_config,
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                parts = data["candidates"][0]["content"]["parts"]
                text_parts = [p["text"] for p in parts if not p.get("thought") and "text" in p]
                if text_parts:
                    return text_parts[-1].strip()
                return parts[0].get("text", "").strip()
            else:
                last_error = f"{response.status_code} from {model}: {response.text}"
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(f"Gemini API request failed: {last_error}")


def _numeric_tokens_in_evidence(evidence: dict) -> set:
    """Collects every numeric value present in the evidence dict, rounded to
    a couple of formats, so we can check the draft doesn't invent new numbers."""
    tokens = set()
    for key, value in evidence.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                if isinstance(sub_v, (int, float)):
                    tokens.add(str(sub_v))
                    tokens.add(str(round(sub_v, 2)))
                    tokens.add(str(round(sub_v, 3)))
                    tokens.add(str(int(round(sub_v * 100))))
                    tokens.add(str(round(sub_v * 100, 1)))
        elif isinstance(value, (int, float)):
            tokens.add(str(value))
            tokens.add(str(round(value, 2)))
            tokens.add(str(round(value, 3)))
            # Add both floor, ceil, round for percentages (e.g. 0.945 -> 94, 95, 94.5)
            pct = value * 100
            tokens.add(str(int(round(pct))))
            tokens.add(str(int(pct)))
            tokens.add(str(int(pct) + 1))
            tokens.add(str(round(pct, 1)))
            tokens.add(str(round(pct, 2)))
    return tokens


def verify_explanation(draft: str, evidence: dict) -> bool:
    """
    A deliberately simple, transparent check (see Part 9 - this is a real
    limitation, not a fully solved problem): every numeric figure quoted in
    the draft must appear somewhere in the evidence dict. This catches the
    most damaging failure mode - the model inventing a percentage or score -
    even though it won't catch every possible subtle misdescription.
    """
    evidence_numbers = _numeric_tokens_in_evidence(evidence)
    numbers_in_draft = re.findall(r"\d+\.?\d*", draft)

    for number in numbers_in_draft:
        # Ignore structural counters like 1, 2, 3 (e.g. "2 factors", "top 2 signals")
        if number in ("1", "2", "3") and number not in evidence_numbers:
            continue
        if number not in evidence_numbers:
            print(f"[Verification Info] Token '{number}' in draft was not found in evidence numbers: {evidence_numbers}")
            return False
    return True


def fallback_template(evidence: dict) -> str:
    """Deterministic, guaranteed-grounded explanation built from the top SHAP-ranked evidence."""
    top_fields = evidence.get("top_evidence", [])
    parts = []
    for field in top_fields:
        value = evidence[field]["value"] if isinstance(evidence[field], dict) else evidence[field]
        parts.append(f"{field.replace('_', ' ')} of {value}")
    joined = " and ".join(parts) if parts else "the combined evidence signals"
    media_subject = "Video classified" if evidence.get("media_type") == "video" else "Classified"
    return (
        f"{media_subject} as {evidence['verdict']} with {evidence['confidence']*100:.0f}% "
        f"confidence, based on {joined}."
    )


def generate_explanation(evidence: dict) -> dict:
    messages = build_messages(evidence)
    try:
        draft = call_gemini(messages)
    except Exception as e:
        return {"explanation": fallback_template(evidence), "source": "fallback", "reason": str(e)}

    print(f"\n[LLM Generated Draft]:\n\"{draft}\"\n")

    if verify_explanation(draft, evidence):
        return {"explanation": draft, "source": "llm_verified"}
    else:
        return {"explanation": fallback_template(evidence), "source": "fallback", "reason": "verification_failed"}


if __name__ == "__main__":
    example_evidence = {
        "verdict": "AI-generated",
        "confidence": 0.945,
        "fft_anomaly_score": {"value": 0.79, "contribution": 0.36},
        "clip_max_generator_similarity": {"value": 0.81, "contribution": 2.05},
        "top_evidence": ["clip_max_generator_similarity", "fft_anomaly_score"],
    }
    print(generate_explanation(example_evidence))
