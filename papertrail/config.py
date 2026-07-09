"""Configuration, storage paths, LLM client setup, and structured-output helpers."""
import json
import logging
import os
import pathlib
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the working directory, if present (see .env.example).
# Real environment variables take precedence over the file.
load_dotenv()

# ChromaDB telemetry is broken in this version pairing (posthog signature
# mismatch) and just spams the log with errors — turn it off.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("papertrail")

# ── Storage paths ─────────────────────────────────────────────────────────────
# Persistent storage path. HF Spaces persistent storage is /data; fall back to ./state locally.
_STATE_CANDIDATES = [pathlib.Path(p) for p in [os.getenv("STATE_DIR", ""), "/data", "./state"] if p]
STATE_DIR = next((p for p in _STATE_CANDIDATES if p.parent.exists() and os.access(p.parent, os.W_OK)), pathlib.Path("./state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
LEGACY_STATE_FILE = STATE_DIR / "state.pkl"
CHROMA_DIR = STATE_DIR / "chroma"
UPLOAD_DIR = str(STATE_DIR / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"State directory: {STATE_DIR}")

# ── LLM client — Groq (OpenAI-compatible API), with optional Gemini fallback ──
_GROQ_KEY = os.getenv("GROQ_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if _GROQ_KEY:
    client = OpenAI(api_key=_GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    _DEFAULT_MODEL = "llama-3.3-70b-versatile"
elif _GEMINI_KEY:
    client = OpenAI(api_key=_GEMINI_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    _DEFAULT_MODEL = "gemini-2.5-flash"
else:
    client = None
    _DEFAULT_MODEL = ""

# Two model lanes: a fast cheap one for classification/extraction, a strong one for synthesis.
MODEL_FAST    = os.getenv("LLM_MODEL_FAST",    os.getenv("GROQ_MODEL", _DEFAULT_MODEL))
MODEL_QUALITY = os.getenv("LLM_MODEL_QUALITY", os.getenv("GROQ_MODEL", _DEFAULT_MODEL))
MODEL = MODEL_QUALITY  # back-compat alias


# ── Rate-limit retry helper ────────────────────────────────────────────────────
def _api_call_with_retry(fn, *args, max_retries: int = 4, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in ("429", "rate limit", "quota", "resource_exhausted", "too many requests"))
            if is_rate_limit and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5 → 10 → 20 → 40 s
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}), retrying in {wait}s…")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


# ── Structured output helper (json_object mode, model-agnostic) ──────────────
def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from a raw LLM response."""
    if not text:
        return "{}"
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Take the largest balanced {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_structured(messages: list, response_model, model: str = None):
    """Call chat completion in json_object mode and parse into a Pydantic model.

    Tries `response_format=json_object` first; if that fails (e.g. the model
    emitted preamble or the provider rejects the output), retries without the
    format constraint and salvages the JSON object from the response.
    """
    schema = response_model.model_json_schema()
    primer = {
        "role": "system",
        "content": (
            "You MUST return exactly one JSON object that conforms to this JSON Schema. "
            "Output the JSON object and NOTHING else — no prose, no markdown fences, "
            "no commentary before or after. Start the response with `{` and end with `}`.\n\n"
            f"Schema:\n{json.dumps(schema)}"
        ),
    }
    msgs = [primer, *messages]
    target_model = model or MODEL_QUALITY

    # Attempt 1: strict json_object mode
    try:
        completion = _api_call_with_retry(
            client.chat.completions.create,
            model=target_model,
            messages=msgs,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        return response_model.model_validate_json(content)
    except Exception as e:
        err = str(e).lower()
        # Bubble rate limits straight up; only fall through on JSON-validation errors.
        if any(k in err for k in ("429", "rate limit", "quota", "resource_exhausted")):
            raise
        logger.warning(f"Strict JSON mode failed ({e.__class__.__name__}); retrying with salvage parse")

    # Attempt 2: free-form, then salvage JSON
    completion = _api_call_with_retry(
        client.chat.completions.create,
        model=target_model,
        messages=msgs,
    )
    raw = completion.choices[0].message.content or "{}"
    return response_model.model_validate_json(_extract_json_object(raw))
