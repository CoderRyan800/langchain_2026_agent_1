"""
Tests for the health analysis module: prompt construction and response parsing.
No LLM or CLIP calls — entirely offline.
"""

import pytest
from litterbox.health import build_health_prompt, parse_health_response, HEALTH_PROMPT


# ---------------------------------------------------------------------------
# build_health_prompt
# ---------------------------------------------------------------------------

class TestBuildHealthPromptNoSensors:
    def test_returns_string(self):
        assert isinstance(build_health_prompt(), str)

    def test_module_constant_matches_no_sensor_call(self):
        assert build_health_prompt() == HEALTH_PROMPT

    def test_contains_format_instruction(self):
        assert "CONCERNS_PRESENT" in build_health_prompt()

    def test_contains_description_field(self):
        assert "DESCRIPTION" in build_health_prompt()

    def test_contains_owner_summary_field(self):
        assert "OWNER_SUMMARY" in build_health_prompt()

    def test_contains_veterinarian_disclaimer(self):
        assert "veterinarian" in build_health_prompt().lower()

    def test_describes_image_comparison_task(self):
        prompt = build_health_prompt()
        assert "Image 1" in prompt and "Image 2" in prompt

    def test_no_sensor_section_when_no_data(self):
        prompt = build_health_prompt()
        assert "sensor data" not in prompt.lower()


class TestBuildHealthPromptWithSensors:
    def test_cat_weight_grams_in_prompt(self):
        assert "4200" in build_health_prompt(cat_weight_g=4200.0)

    def test_cat_weight_kg_in_prompt(self):
        # 4200 g → 4.200 kg
        assert "4.200" in build_health_prompt(cat_weight_g=4200.0)

    def test_waste_weight_in_prompt(self):
        assert "68" in build_health_prompt(waste_weight_g=68.0)

    def test_weight_pre_in_prompt(self):
        assert "5800" in build_health_prompt(weight_pre_g=5800.0)

    def test_weight_entry_in_prompt(self):
        assert "10050" in build_health_prompt(weight_entry_g=10050.0)

    def test_weight_exit_in_prompt(self):
        assert "5868" in build_health_prompt(weight_exit_g=5868.0)

    def test_ammonia_value_in_prompt(self):
        assert "310" in build_health_prompt(ammonia_peak_ppb=310.0)

    def test_ammonia_label_in_prompt(self):
        assert "NH" in build_health_prompt(ammonia_peak_ppb=310.0)

    def test_methane_value_in_prompt(self):
        assert "55" in build_health_prompt(methane_peak_ppb=55.0)

    def test_methane_label_in_prompt(self):
        assert "CH" in build_health_prompt(methane_peak_ppb=55.0)

    def test_all_sensors_all_present(self):
        prompt = build_health_prompt(
            weight_pre_g=5800.0, weight_entry_g=10050.0, weight_exit_g=5868.0,
            cat_weight_g=4250.0, waste_weight_g=68.0,
            ammonia_peak_ppb=310.0, methane_peak_ppb=55.0,
        )
        for value in ("5800", "10050", "5868", "4250", "68", "310", "55"):
            assert value in prompt, f"Expected {value!r} in prompt"

    def test_partial_sensors_omits_missing(self):
        prompt = build_health_prompt(ammonia_peak_ppb=100.0)
        assert "NH" in prompt
        assert "CH" not in prompt      # methane not provided
        assert "5800" not in prompt    # no weight provided

    def test_sensor_section_appears_before_format_instructions(self):
        prompt = build_health_prompt(cat_weight_g=4200.0)
        sensor_idx  = prompt.index("4200")
        format_idx  = prompt.index("CONCERNS_PRESENT")
        assert sensor_idx < format_idx

    def test_format_instructions_always_present_with_sensors(self):
        prompt = build_health_prompt(ammonia_peak_ppb=100.0)
        assert "CONCERNS_PRESENT" in prompt
        assert "DESCRIPTION"      in prompt
        assert "OWNER_SUMMARY"    in prompt

    def test_disclaimer_always_present_with_sensors(self):
        prompt = build_health_prompt(cat_weight_g=4000.0)
        assert "veterinarian" in prompt.lower()

    def test_sensor_context_guidance_included(self):
        prompt = build_health_prompt(ammonia_peak_ppb=200.0)
        assert "ammonia" in prompt.lower()

    def test_zero_values_appear_in_prompt(self):
        prompt = build_health_prompt(cat_weight_g=0.0)
        assert "0.0" in prompt or "0 g" in prompt


# ---------------------------------------------------------------------------
# parse_health_response
# ---------------------------------------------------------------------------

class TestParseHealthResponse:
    def test_yes_is_anomalous(self):
        anomalous, _ = parse_health_response("CONCERNS_PRESENT: yes\nDESCRIPTION: x")
        assert anomalous is True

    def test_no_is_not_anomalous(self):
        anomalous, _ = parse_health_response("CONCERNS_PRESENT: no\nDESCRIPTION: x")
        assert anomalous is False

    def test_uppercase_yes(self):
        anomalous, _ = parse_health_response("CONCERNS_PRESENT: YES")
        assert anomalous is True

    def test_uppercase_no(self):
        anomalous, _ = parse_health_response("CONCERNS_PRESENT: NO")
        assert anomalous is False

    def test_mixed_case_key(self):
        anomalous, _ = parse_health_response("concerns_present: no")
        assert anomalous is False

    def test_malformed_response_defaults_false(self):
        anomalous, _ = parse_health_response("This is not a structured response.")
        assert anomalous is False

    def test_empty_string_defaults_false(self):
        anomalous, _ = parse_health_response("")
        assert anomalous is False

    def test_full_text_preserved(self):
        text = "CONCERNS_PRESENT: no\nDESCRIPTION: fine\nOWNER_SUMMARY: all good"
        _, notes = parse_health_response(text)
        assert notes == text

    def test_anomaly_text_preserved(self):
        text = "CONCERNS_PRESENT: yes\nDESCRIPTION: blood visible"
        anomalous, notes = parse_health_response(text)
        assert anomalous is True
        assert "blood visible" in notes

    def test_returns_tuple_of_bool_and_str(self):
        result = parse_health_response("CONCERNS_PRESENT: no")
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_yes_takes_priority_over_later_no(self):
        # First occurrence wins
        text = "CONCERNS_PRESENT: yes\nCONCERNS_PRESENT: no"
        anomalous, _ = parse_health_response(text)
        assert anomalous is True
