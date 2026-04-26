"""
Free-text → structured CustomerFeatures.

Primary path: Google Gen AI SDK (Gemini) with a Pydantic response_schema.
Fallback path: a deterministic regex/keyword parser used when no API key is set
or when the Gemini call fails for any reason. The fallback is intentionally
simple — it covers the common phrasings most demo prompts use and falls through
to defaults for anything ambiguous.

The merged result always contains every field, with a per-field provenance map:
"extracted" (Gemini), "regex" (fallback), or "defaulted" (training-data median).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EXTRACTOR_MODEL = "gemini-2.5-flash"

EXTRACTOR_SYSTEM_PROMPT = """You are a feature-extraction assistant for a residential
solar-and-battery sizing model. The user describes a household in free text.
Extract whichever of the following fields you can confidently infer; leave the
rest as null. Do not guess — only fill a field when the text clearly supports it.

Fields:
- energy_demand_wh: Annual household electricity consumption in WATT-HOURS (Wh).
  Convert kWh → Wh by ×1000, MWh → Wh by ×1_000_000.
  Examples: "5 MWh/year" → 5000000; "4500 kWh annual" → 4500000.
- has_ev: 1 if the household has an electric vehicle, 0 if explicitly stated none, else null.
- has_solar: 1 if existing solar panels, 0 if explicitly none, else null.
- has_storage: 1 if existing battery storage, 0 if explicitly none, else null.
- has_wallbox: 1 if a wallbox / EV charger is present, 0 if explicitly none, else null.
  If the user mentions an EV and home charging without naming the wallbox, set has_wallbox=1.
- house_size_sqm: Building floor area in square metres.
  Convert sqft → sqm by ×0.0929 if the user gave imperial units.
- heating_existing_electricity_demand_kwh: Annual electricity used for heating, in kWh.
  Set to 0 only if the user explicitly says they have no electric heating.

Rules:
- Do NOT invent annual demand from house size alone.
- If a field is genuinely not implied by the text, return null — defaults are filled in downstream.
"""


class PartialCustomerFeatures(BaseModel):
    """
    Anything we could pull out of the prompt. Null = unknown, fill from defaults.

    Schema is kept loose because Gemini's `response_schema` rejects several
    standard JSON Schema keywords (`exclusiveMinimum`, `minimum`, `maximum`,
    `additionalProperties`). The downstream CustomerFeatures model re-validates
    everything before it reaches the inference pipeline.
    """

    energy_demand_wh: Optional[float] = Field(default=None, description="Annual demand in Wh")
    has_ev: Optional[int] = Field(default=None, description="0 or 1")
    has_solar: Optional[int] = Field(default=None, description="0 or 1")
    has_storage: Optional[int] = Field(default=None, description="0 or 1")
    has_wallbox: Optional[int] = Field(default=None, description="0 or 1")
    house_size_sqm: Optional[float] = Field(default=None, description="Building area in m²")
    heating_existing_electricity_demand_kwh: Optional[float] = Field(default=None, description="kWh/year")


# Defaults derived from training-data medians/modes (data/reonic_training_ready.csv, n=1126).
DEFAULTS: dict[str, float | int] = {
    "energy_demand_wh": 4_500_000.0,
    "has_ev": 0,
    "has_solar": 0,
    "has_storage": 0,
    "has_wallbox": 0,
    "house_size_sqm": 150.0,
    "heating_existing_electricity_demand_kwh": 0.0,
}


def _api_key() -> Optional[str]:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_AI_API_KEY")
    )


def is_configured() -> bool:
    """True iff a Gemini API key is set. Always-true now that we have a fallback,
    so callers can simply rely on extract_features() instead of gating."""
    return True


def _gemini_extract(prompt: str) -> dict:
    """Call Gemini and return a dict mirroring PartialCustomerFeatures (None = unknown)."""
    client = genai.Client(api_key=_api_key())
    response = client.models.generate_content(
        model=EXTRACTOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTOR_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=PartialCustomerFeatures,
            temperature=0.0,
            max_output_tokens=512,
        ),
    )
    parsed: PartialCustomerFeatures = response.parsed
    return parsed.model_dump()


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------

_NUM = r"(\d+(?:[.,]\d+)?)"  # matches "4500", "4.5", "4,500"


def _to_float(s: str) -> float:
    return float(s.replace(",", "").replace(" ", ""))


def _has_negation(text: str, term: str) -> bool:
    """True if `term` appears with a negation in front of it (no/not/without)."""
    return bool(re.search(rf"\b(?:no|not|without|dont|don'?t|none of)\s+(?:\w+\s+){{0,3}}{term}", text))


def _regex_extract(prompt: str) -> dict:
    """
    Deterministic best-effort extractor. Returns a dict with the same shape as
    PartialCustomerFeatures (None = couldn't infer). Intentionally conservative —
    when in doubt, leaves a field as None so the default fills in.
    """
    text = prompt.lower()

    out: dict = {
        "energy_demand_wh": None,
        "has_ev": None,
        "has_solar": None,
        "has_storage": None,
        "has_wallbox": None,
        "house_size_sqm": None,
        "heating_existing_electricity_demand_kwh": None,
    }

    # Demand: prefer MWh > kWh > Wh, take the first match.
    m = re.search(rf"{_NUM}\s*mwh", text)
    if m:
        out["energy_demand_wh"] = _to_float(m.group(1)) * 1_000_000
    else:
        m = re.search(rf"{_NUM}\s*kwh(?!\W*(?:heating|heat))", text)
        if m:
            out["energy_demand_wh"] = _to_float(m.group(1)) * 1_000
        else:
            m = re.search(rf"{_NUM}\s*wh\b", text)
            if m:
                out["energy_demand_wh"] = _to_float(m.group(1))

    # House size: m², sqm, m2, or "<n> square metres". sqft → ×0.0929.
    m = re.search(rf"{_NUM}\s*(?:m²|m\^?2|sqm|sq\.?\s*m|square\s*met(?:er|re)s?)", text)
    if m:
        out["house_size_sqm"] = _to_float(m.group(1))
    else:
        m = re.search(rf"{_NUM}\s*(?:sqft|sq\.?\s*ft|square\s*feet)", text)
        if m:
            out["house_size_sqm"] = round(_to_float(m.group(1)) * 0.0929, 1)

    # Heating: explicit "<n> kWh ... heating".
    m = re.search(rf"heating[^.]*?{_NUM}\s*kwh", text) or re.search(rf"{_NUM}\s*kwh[^.]*?heating", text)
    if m:
        out["heating_existing_electricity_demand_kwh"] = _to_float(m.group(1))
    elif re.search(r"no\s+(?:electric\s+)?heating", text):
        out["heating_existing_electricity_demand_kwh"] = 0.0

    # Boolean flags. Order matters: check negation first so "no EV" wins over "EV".
    flags: list[tuple[str, str]] = [
        ("has_ev", r"\bevs?\b|electric\s+vehicles?|electric\s+cars?"),
        ("has_solar", r"\bsolar\b|\bpv\b|photovoltaic"),
        ("has_storage", r"\bbatter(?:y|ies)\b|storage|powerwall"),
        ("has_wallbox", r"\bwall\s*box\b|wallbox|ev\s*charger|home\s*charg"),
    ]
    for field, pattern in flags:
        if re.search(pattern, text):
            # Find an actual match-position negation: "no EV", "without solar", etc.
            negated = bool(re.search(rf"\b(?:no|not|without|don'?t\s+have|none)\s+(?:\w+\s+){{0,3}}(?:{pattern})", text))
            out[field] = 0 if negated else 1

    # "EV charging at home" → wallbox=1 if not already set.
    if out["has_wallbox"] is None and out["has_ev"] == 1 and re.search(r"charg(?:e|ing)\s+at\s+home|home\s+charg", text):
        out["has_wallbox"] = 1

    return out


def extract_features(prompt: str) -> tuple[dict, dict]:
    """
    Run the extractor on a free-text prompt.

    Tries Gemini first when GEMINI_API_KEY is set; falls back to a deterministic
    regex parser on any failure (or when no key is configured). Always returns a
    fully-populated feature dict and a per-field provenance map.

    provenance values: "extracted" (Gemini), "regex" (fallback), "defaulted".
    """
    extracted_dict: dict = {}
    source: str = "regex"  # default when Gemini is skipped or fails

    if _api_key():
        try:
            extracted_dict = _gemini_extract(prompt)
            source = "extracted"
        except Exception as e:
            logger.warning("Gemini extraction failed (%s); falling back to regex.", e)
            extracted_dict = _regex_extract(prompt)
    else:
        logger.info("No GEMINI_API_KEY set; using regex extractor.")
        extracted_dict = _regex_extract(prompt)

    features: dict[str, float | int] = {}
    provenance: dict[str, str] = {}
    for field, default in DEFAULTS.items():
        value = extracted_dict.get(field)
        if value is None:
            features[field] = default
            provenance[field] = "defaulted"
        else:
            features[field] = value
            provenance[field] = source

    logger.info("Extraction (%s): provenance=%s", source, provenance)
    return features, provenance
