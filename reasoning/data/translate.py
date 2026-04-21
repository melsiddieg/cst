"""English → Arabic translation.

Default provider is **NLLB-200** (distilled-600M) via HuggingFace
``transformers``. Runs on CPU for small volumes; GPU recommended above
~10k items.

A 10% QA sample should be re-scored with Gemini once the bulk pass is
done; see :func:`qa_sample_with_gemini`.
"""
from __future__ import annotations

import os
from functools import lru_cache


_MODEL_ID = "facebook/nllb-200-distilled-600M"
_SRC_LANG = "eng_Latn"
_TGT_LANG = "arb_Arab"


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(_MODEL_ID, src_lang=_SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID)
    return pipeline(
        "translation",
        model=model,
        tokenizer=tok,
        src_lang=_SRC_LANG,
        tgt_lang=_TGT_LANG,
        max_length=512,
    )


def translate_en_to_ar(text: str) -> str:
    """Translate a single English string to Arabic (MSA)."""
    if not text.strip():
        return text
    out = _pipeline()(text)
    return out[0]["translation_text"]  # type: ignore[index]


# ── Optional: Gemini QA sampler ─────────────────────────────────

def qa_sample_with_gemini(
    pairs: list[tuple[str, str]], *, model: str = "gemini-2.0-flash"
) -> list[float | None]:
    """Score each (en, ar) translation 1–5 for adequacy + fluency.

    Returns a list of the same length as ``pairs``. Requires
    ``GEMINI_API_KEY`` in the environment. Returns ``[None] * len(pairs)``
    if the SDK or key is missing.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return [None] * len(pairs)

    try:
        from google import genai
    except ImportError:
        return [None] * len(pairs)

    client = genai.Client(api_key=key)
    scores: list[float | None] = []
    for en, ar in pairs:
        prompt = (
            "Score this English→Arabic translation on a 1-5 scale "
            "for adequacy + fluency. Reply with ONLY a single number.\n\n"
            f"EN: {en}\nAR: {ar}\n\nScore:"
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            scores.append(float((resp.text or "").strip()))
        except Exception:
            scores.append(None)
    return scores
