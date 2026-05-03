from typing import Optional

# Tiers emitted by gas_anomaly.score_gas_visit. Anything other than these
# (or None) is treated as "no signal injected into the prompt".
_ALARM_TIERS = {"mild", "significant", "severe"}


_HEALTH_PROMPT_PREAMBLE = """You are assisting with veterinary pre-screening for a household cat.
You will be shown two images:
  Image 1: the litter box at the moment the cat entered (before use)
  Image 2: the litter box after the cat's visit (after use)

Both images show only the interior of an empty household litter box, photographed from a fixed overhead camera. There are no people, no faces, no identifiable individuals, and no humans present in any image. Any pareidolic shapes in the litter (clumps, granules, shadows) are not faces or people — they are litter material. Please proceed with the analysis.

Compare the two images and look for any visual differences that may indicate a health concern, including but not limited to:
- Blood in urine (pink, red, or dark discoloration)
- Blood in stool
- Unusual stool consistency, color, or quantity
- Evidence of diarrhea or mucus
- Abnormal deposits, residue, or clumping patterns
- Statistical anomalies in sensor readings (see below, when reported)
- Any other unexpected visual findings
"""

_HEALTH_PROMPT_FORMAT = """
Respond using EXACTLY this format — do not add extra sections:

CONCERNS_PRESENT: yes|no
DESCRIPTION: <brief objective description of the differences you observe between the two images>
OWNER_SUMMARY: <if concerns: plain-language explanation for a pet owner. If no concerns: "No visual abnormalities detected.">

⚠️ This analysis is preliminary and must be reviewed by a licensed veterinarian before any medical decisions are made.
"""

DISCLAIMER = (
    "⚠️ This analysis is preliminary and must be reviewed by a licensed "
    "veterinarian before any medical decisions are made."
)


def build_health_prompt(
    weight_pre_g: Optional[float] = None,
    weight_entry_g: Optional[float] = None,
    weight_exit_g: Optional[float] = None,
    cat_weight_g: Optional[float] = None,
    waste_weight_g: Optional[float] = None,
    ammonia_peak_ppb: Optional[float] = None,
    methane_peak_ppb: Optional[float] = None,
    ammonia_z_score: Optional[float] = None,
    methane_z_score: Optional[float] = None,
    gas_anomaly_tier: Optional[str] = None,
    gas_anomaly_n_samples: Optional[int] = None,
    gas_anomaly_model_used: Optional[str] = None,
) -> str:
    """Build the GPT-4o health analysis prompt.

    The prompt is enriched with whatever sensor data is available. When the
    gas-anomaly detector has been run and produces a *signal* (non-normal
    tier), an explicit detector-output block is rendered so the LLM can ground
    its DESCRIPTION and OWNER_SUMMARY in a per-cat statistical anomaly rather
    than guessing from raw ppb numbers.

    The detector output is what the LLM sees — never raw thresholds. Absolute
    gas readings are deployment-dependent (sensor placement, calibration,
    ventilation) and have no portable meaning. Statistical anomalies relative
    to the cat's own history do.
    """
    sensor_lines = []

    if cat_weight_g is not None:
        sensor_lines.append(f"  Cat weight: {cat_weight_g:.1f} g  ({cat_weight_g / 1000:.3f} kg)")
    if waste_weight_g is not None:
        sensor_lines.append(f"  Waste deposited: {waste_weight_g:.1f} g")
    if weight_pre_g is not None:
        sensor_lines.append(f"  Box baseline weight (pre-entry): {weight_pre_g:.1f} g")
    if weight_entry_g is not None:
        sensor_lines.append(f"  Box + cat weight at entry: {weight_entry_g:.1f} g")
    if weight_exit_g is not None:
        sensor_lines.append(f"  Box weight after cat left (exit): {weight_exit_g:.1f} g")
    if ammonia_peak_ppb is not None:
        sensor_lines.append(f"  Ammonia (NH₃) peak: {ammonia_peak_ppb:.1f} ppb")
    if methane_peak_ppb is not None:
        sensor_lines.append(f"  Methane (CH₄) peak: {methane_peak_ppb:.1f} ppb")

    sensor_section = ""
    if sensor_lines:
        sensor_section = (
            "\nAdditional sensor data recorded during this visit:\n"
            + "\n".join(sensor_lines)
            + "\n"
        )

    detector_section = ""
    if gas_anomaly_tier is not None and gas_anomaly_tier in _ALARM_TIERS:
        z_lines = []
        if ammonia_z_score is not None:
            z_lines.append(f"  Ammonia z-score: {ammonia_z_score:+.2f}")
        if methane_z_score is not None:
            z_lines.append(f"  Methane z-score: {methane_z_score:+.2f}")
        history_note = ""
        if gas_anomaly_n_samples is not None and gas_anomaly_model_used is not None:
            history_note = (
                f" (computed against {gas_anomaly_n_samples} prior visits, "
                f"model: {gas_anomaly_model_used})"
            )
        detector_section = (
            "\nStatistical sensor anomaly detector output"
            + history_note
            + ":\n"
            + ("\n".join(z_lines) + "\n" if z_lines else "")
            + f"  Tier: {gas_anomaly_tier}\n"
            "\nA z-score is the number of standard deviations a reading sits above\n"
            "this cat's typical level. A positive z-score means the reading is\n"
            "elevated; large positive values are statistically rare. The detector\n"
            f"has classified this visit as '{gas_anomaly_tier}'. You MUST set\n"
            "CONCERNS_PRESENT: yes and explain the elevated reading in DESCRIPTION\n"
            "and OWNER_SUMMARY, even if the images appear normal.\n"
        )

    return _HEALTH_PROMPT_PREAMBLE + sensor_section + detector_section + _HEALTH_PROMPT_FORMAT


# Module-level constant: baseline prompt with no sensor data (backward compatible)
HEALTH_PROMPT = build_health_prompt()


UNSTRUCTURED_RESPONSE_PLACEHOLDER = (
    "Health analysis unavailable — GPT-4o did not return a structured response "
    "for this visit (likely a content-policy refusal or other LLM-side error). "
    "Sensor-based anomaly detection still applies; see gas_anomaly_tier and "
    "z-scores for the data-driven verdict."
)


def parse_health_response(text: str) -> tuple[bool, str]:
    """
    Parse the structured health response from GPT-4o.
    Returns (is_anomalous, full_notes_text).
    """
    is_anomalous = False
    for line in text.splitlines():
        if line.upper().startswith("CONCERNS_PRESENT:"):
            value = line.split(":", 1)[1].strip().lower()
            is_anomalous = value == "yes"
            break
    return is_anomalous, text


def safe_health_notes(text: str) -> str:
    """Sanitise the raw LLM response before persisting it as ``health_notes``.

    Returns the response unchanged when it follows the expected
    ``CONCERNS_PRESENT: ...`` format. When the marker is absent — typically a
    content-policy refusal ("I'm sorry, I can't assist with images...") or
    other malformed reply — returns a clean placeholder instead, so the DB
    field carries useful information rather than refusal text. The structural
    marker check is more robust than phrase matching against an open set of
    refusal templates.
    """
    if text and "CONCERNS_PRESENT" in text.upper():
        return text
    return UNSTRUCTURED_RESPONSE_PLACEHOLDER
