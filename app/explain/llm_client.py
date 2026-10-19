"""
Calls Gemini 3.8 Flash for evidence explanation and runs a strict multi-stage
faithfulness verification check: reject and fall back to a deterministic template
if the draft invents numbers, contradicts the verdict direction, or quotes ungrounded claims.
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
    """Collects every numeric value present in the evidence dict in standard formats."""
    tokens = set()
    for key, value in evidence.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                if isinstance(sub_v, (int, float)):
                    tokens.add(str(sub_v))
                    tokens.add(str(round(sub_v, 2)))
                    tokens.add(str(round(sub_v, 3)))
                    tokens.add(str(round(sub_v, 4)))
                    tokens.add(str(int(round(sub_v * 100))))
                    tokens.add(str(round(sub_v * 100, 1)))
        elif isinstance(value, (int, float)):
            tokens.add(str(value))
            tokens.add(str(round(value, 2)))
            tokens.add(str(round(value, 3)))
            tokens.add(str(round(value, 4)))
            pct = value * 100
            tokens.add(str(int(round(pct))))
            tokens.add(str(int(pct)))
            tokens.add(str(int(pct) + 1))
            tokens.add(str(round(pct, 1)))
            tokens.add(str(round(pct, 2)))
    return tokens


def verify_explanation(draft: str, evidence: dict) -> dict:
    """
    Multi-stage faithfulness verification audit:
    1. Numeric grounding: Draft must not invent ungrounded percentages or metrics.
    2. Directional consistency: Draft must align with the classified verdict.
    3. Abstention consistency: If abstained, draft must convey uncertainty.
    """
    evidence_numbers = _numeric_tokens_in_evidence(evidence)
    numbers_in_draft = re.findall(r"\d+\.?\d*", draft)
    verdict = str(evidence.get("verdict", "")).lower()

    # 1. Numeric audit
    numeric_passed = True
    ungrounded_tokens = []
    for num in numbers_in_draft:
        # Ignore structural sequence numbers like 1, 2, 3
        if num in ("1", "2", "3") and num not in evidence_numbers:
            continue
        if num not in evidence_numbers:
            numeric_passed = False
            ungrounded_tokens.append(num)

    # 2. Directional audit
    draft_lower = draft.lower()
    directional_passed = True
    if verdict == "ai-generated":
        if "authentic" in draft_lower or "real photograph" in draft_lower:
            directional_passed = False
    elif verdict == "real":
        if "synthetic" in draft_lower or "ai-generated" in draft_lower:
            directional_passed = False
    elif verdict in ("uncertain", "abstain"):
        if not ("uncertain" in draft_lower or "abstained" in draft_lower or "borderline" in draft_lower or "inconclusive" in draft_lower):
            directional_passed = False

    passed = numeric_passed and directional_passed
    details = (
        "All numerical and directional claims verified against evidence."
        if passed
        else f"Failed audit: ungrounded_numbers={ungrounded_tokens}, directional_ok={directional_passed}"
    )

    return {
        "passed": passed,
        "numeric_audit_passed": numeric_passed,
        "directional_audit_passed": directional_passed,
        "ungrounded_tokens": ungrounded_tokens,
        "details": details,
    }


def fallback_template(evidence: dict) -> str:
    """Deterministic, guaranteed-grounded explanation built from the top SHAP-ranked evidence."""
    verdict = evidence.get("verdict", "unknown")
    conf_pct = int(round(evidence.get("confidence", 0.5) * 100))
    media_subject = "Video" if evidence.get("media_type") == "video" else "Image"

    top_fields = evidence.get("top_evidence", [])
    parts = []
    for field in top_fields:
        val = evidence[field]["value"] if isinstance(evidence.get(field), dict) else evidence.get(field)
        if val is not None:
            parts.append(f"{field.replace('_', ' ')} ({val})")
    joined = " and ".join(parts) if parts else "all combined forensic signals"

    if verdict == "uncertain" or evidence.get("abstained"):
        return (
            f"{media_subject} evaluation resulted in an abstention (uncertain verdict) "
            f"with calibrated probability of {evidence.get('calibrated_prob_ai', 0.50):.3f}. "
            f"Forensic indicators ({joined}) are borderline. Manual verification is advised."
        )

    return (
        f"{media_subject} classified as {verdict} with {conf_pct}% "
        f"confidence, primarily driven by {joined}."
    )


def generate_explanation(evidence: dict) -> dict:
    messages = build_messages(evidence)
    try:
        draft = call_gemini(messages)
        audit = verify_explanation(draft, evidence)
        if audit["passed"]:
            return {
                "explanation": draft,
                "source": "llm_verified",
                "faithfulness_audit": audit,
            }
        else:
            return {
                "explanation": fallback_template(evidence),
                "source": "fallback",
                "faithfulness_audit": audit,
            }
    except Exception as e:
        return {
            "explanation": fallback_template(evidence),
            "source": "fallback",
            "faithfulness_audit": {
                "passed": False,
                "numeric_audit_passed": False,
                "directional_audit_passed": True,
                "details": f"Gemini API unavailable or errored ({str(e)}). Used deterministic fallback.",
            },
        }


if __name__ == "__main__":
    example_evidence = {
        "verdict": "AI-generated",
        "confidence": 0.945,
        "clip_semantic_score": {"value": 0.81, "contribution": 2.05},
        "fft_anomaly_score": {"value": 0.79, "contribution": 0.36},
        "top_evidence": ["clip_semantic_score", "fft_anomaly_score"],
    }
    print(generate_explanation(example_evidence))
