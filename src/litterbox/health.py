HEALTH_PROMPT = """You are assisting with veterinary pre-screening for a household cat.
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
