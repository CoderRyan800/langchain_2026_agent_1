from typing import Optional

_HEALTH_PROMPT_PREAMBLE = """You are assisting with veterinary pre-screening for a household cat.
You will be shown two images:
  Image 1: the litter box at the moment the cat entered (before use)
  Image 2: the litter box after the cat's visit (after use)

Compare the two images and look for any visual differences that may indicate a health concern, including but not limited to:
- Blood in urine (pink, red, or dark discoloration)
- Blood in stool
- Unusual stool consistency, color, or quantity
- Evidence of diarrhea or mucus
- Abnormal deposits, residue, or clumping patterns
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
) -> str:
    """Build the GPT-4o health analysis prompt, enriched with any available sensor data."""
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
        sensor_lines.append(f"  Ammonia (NH\u2083) peak: {ammonia_peak_ppb:.1f} ppb")
    if methane_peak_ppb is not None:
        sensor_lines.append(f"  Methane (CH\u2084) peak: {methane_peak_ppb:.1f} ppb")

    if sensor_lines:
        sensor_section = (
            "\nAdditional sensor data recorded during this visit:\n"
            + "\n".join(sensor_lines)
            + "\n\nConsider these readings alongside the visual evidence when assessing "
            "potential health concerns. Elevated ammonia may indicate increased or "
            "concentrated urine output. Significant weight changes over time may indicate "
            "health trends worth monitoring.\n"
        )
    else:
        sensor_section = ""

    return _HEALTH_PROMPT_PREAMBLE + sensor_section + _HEALTH_PROMPT_FORMAT


# Module-level constant: baseline prompt with no sensor data (backward compatible)
HEALTH_PROMPT = build_health_prompt()


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
