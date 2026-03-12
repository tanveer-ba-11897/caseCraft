"""
Comprehensive tests for core.generator module.
Tests: _clean_json_output, _sanitize_test_cases, _deduplicate_test_cases,
       and _clean_test_case_titles.
"""

import json
import pytest
from core import generator


class TestCleanJsonOutput:
    """Tests for _clean_json_output function."""

    def test_removes_think_blocks(self):
        text = '<think>This is thinking</think>{"test": "value"}'
        result = generator._clean_json_output(text)
        assert result == '{"test": "value"}'

    def test_removes_markdown_code_blocks(self):
        text = '```json\n[{"test": "value"}]\n```'
        result = generator._clean_json_output(text)
        assert result == '[{"test": "value"}]'

    def test_removes_markdown_without_language(self):
        text = '```\n{"test": "value"}\n```'
        result = generator._clean_json_output(text)
        assert result == '{"test": "value"}'

    def test_extracts_json_array(self):
        text = 'Here is the output: [{"test": "value"}] Thanks!'
        result = generator._clean_json_output(text)
        assert result == '[{"test": "value"}]'

    def test_extracts_json_object(self):
        text = 'Generated: {"test": "value"} end'
        result = generator._clean_json_output(text)
        assert result == '{"test": "value"}'

    def test_prefers_earlier_delimiter(self):
        # Array starts first
        text = '[ {"nested": true} ]'
        result = generator._clean_json_output(text)
        assert result.startswith('[')

    def test_handles_complex_think_blocks(self):
        text = '<think>\nLet me think about this...\nDone.\n</think>\n[{"case": 1}]'
        result = generator._clean_json_output(text)
        assert '<think>' not in result
        assert result == '[{"case": 1}]'

    def test_handles_no_json(self):
        text = 'This is just plain text'
        result = generator._clean_json_output(text)
        assert result == 'This is just plain text'

    def test_handles_empty_input(self):
        result = generator._clean_json_output('')
        assert result == ''

    def test_repairs_trailing_comma_in_array(self):
        text = '[{"test": "value"},]'
        result = generator._clean_json_output(text)
        parsed = json.loads(result)
        assert parsed == [{"test": "value"}]

    def test_repairs_trailing_comma_in_object(self):
        text = '{"key": "value", "key2": "value2",}'
        result = generator._clean_json_output(text)
        parsed = json.loads(result)
        assert parsed == {"key": "value", "key2": "value2"}

    def test_repairs_truncated_json_array(self):
        """Simulates LLM output truncated mid-array (common with token cap)."""
        text = '[{"use_case": "Login", "test_case": "Valid"}, {"use_case": "Logout", "test_ca'
        result = generator._clean_json_output(text)
        parsed = json.loads(result)
        assert len(parsed) >= 1
        assert parsed[0]["use_case"] == "Login"

    def test_repairs_truncated_json_with_trailing_comma(self):
        text = '[{"a": "1"}, {"b": "2"},'
        result = generator._clean_json_output(text)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_repairs_multiple_trailing_commas(self):
        text = '[{"steps": ["Step 1", "Step 2",], "tags": ["tag1",]},]'
        result = generator._clean_json_output(text)
        parsed = json.loads(result)
        assert parsed[0]["steps"] == ["Step 1", "Step 2"]


class TestSanitizeTestCases:
    """Tests for _sanitize_test_cases function."""

    def test_converts_string_expected_results_to_list(self):
        cases = [{"expected_results": "string value"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["expected_results"] == ["string value"]

    def test_coerces_non_string_items_in_expected_results(self):
        cases = [{"expected_results": ["valid", 123, {"nested": True}, "also valid"]}]
        result = generator._sanitize_test_cases(cases)
        # int 123 -> "123", dict -> "nested: True"
        assert result[0]["expected_results"] == ["valid", "123", "nested: True", "also valid"]

    def test_provides_default_use_case(self):
        cases = [{"test_case": "some test"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["use_case"] == "General Use Case"

    def test_provides_default_test_case(self):
        cases = [{"use_case": "some use case"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_case"] == "Untitled Test Case"

    def test_provides_default_priority(self):
        cases = [{}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["priority"] == "medium"

    def test_converts_non_list_preconditions(self):
        cases = [{"preconditions": "single string"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["preconditions"] == ["single string"]

    def test_converts_non_list_steps(self):
        cases = [{"steps": "single step"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["steps"] == ["single step"]

    def test_converts_non_list_tags(self):
        cases = [{"tags": "single tag"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["tags"] == ["single tag"]

    def test_converts_non_dict_test_data(self):
        cases = [{"test_data": [{"key": "user", "value": "admin"}]}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_data"] == {"key": "user", "value": "admin"}

    def test_converts_test_data_values_to_strings(self):
        cases = [{"test_data": {"num": 123, "bool": True, "str": "value"}}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_data"] == {"num": "123", "bool": "True", "str": "value"}

    def test_does_not_modify_original(self):
        original = [{"use_case": "test", "expected_results": "string"}]
        generator._sanitize_test_cases(original)
        # Original should still have the string value
        assert original[0]["expected_results"] == "string"

    def test_coerces_list_of_dicts_in_steps(self):
        """LLMs sometimes return steps as [{"step": 1, "action": "Click"}]."""
        cases = [{"steps": [{"step": 1, "action": "Click login"}, {"step": 2, "action": "Enter password"}]}]
        result = generator._sanitize_test_cases(cases)
        assert all(isinstance(s, str) for s in result[0]["steps"])
        assert len(result[0]["steps"]) == 2

    def test_coerces_list_test_data_to_dict(self):
        """LLMs sometimes return test_data as a list of dicts."""
        cases = [{"test_data": [{"username": "admin"}, {"password": "secret"}]}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_data"] == {"username": "admin", "password": "secret"}

    def test_coerces_empty_dict_in_steps_skipped(self):
        """Empty dicts in list fields should be silently skipped."""
        cases = [{"steps": [{}, "Step 1", {"action": "Click"}]}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["steps"] == ["Step 1", "action: Click"]

    def test_coerces_none_values_in_dict_filtered(self):
        """None values inside dict items should be excluded from coerced strings."""
        cases = [{"steps": [{"step": 1, "note": None, "action": "Click"}]}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["steps"] == ["step: 1, action: Click"]

    def test_coerces_int_and_float_in_list_fields(self):
        """Numeric values in list fields should be stringified."""
        cases = [{"steps": [1, 2.5, "Step 3"]}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["steps"] == ["1", "2.5", "Step 3"]

    def test_test_data_dict_checked_before_list(self):
        """test_data as a plain dict should be handled correctly (not mistaken for list)."""
        cases = [{"test_data": {"user": "admin", "count": 5}}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_data"] == {"user": "admin", "count": "5"}

    def test_test_data_non_dict_non_list_defaults_empty(self):
        """Non-dict/list test_data should default to empty dict."""
        cases = [{"test_data": "invalid"}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_data"] == {}

    def test_test_data_list_with_non_dict_items_ignored(self):
        """Non-dict items in test_data list should be silently ignored."""
        cases = [{"test_data": [{"user": "admin"}, "invalid", 42]}]
        result = generator._sanitize_test_cases(cases)
        assert result[0]["test_data"] == {"user": "admin"}


class TestDeduplicateTestCases:
    """Tests for _deduplicate_test_cases function."""

    def test_removes_exact_duplicates(self):
        cases = [
            {"use_case": "Login", "test_case": "Valid Login", "steps": ["Enter credentials", "Click login"]},
            {"use_case": "Login", "test_case": "Valid Login", "steps": ["Enter credentials", "Click login"]},
        ]
        result = generator._deduplicate_test_cases(cases)
        assert len(result) == 1

    def test_removes_case_insensitive_duplicates(self):
        cases = [
            {"use_case": "LOGIN", "test_case": "VALID LOGIN", "steps": ["ENTER CREDENTIALS"]},
            {"use_case": "login", "test_case": "valid login", "steps": ["enter credentials"]},
        ]
        result = generator._deduplicate_test_cases(cases)
        assert len(result) == 1

    def test_removes_punctuation_normalized_duplicates(self):
        cases = [
            {"use_case": "User Login!", "test_case": "Valid Login.", "steps": ["Step 1..."]},
            {"use_case": "User Login", "test_case": "Valid Login", "steps": ["Step 1"]},
        ]
        result = generator._deduplicate_test_cases(cases)
        assert len(result) == 1

    def test_preserves_first_occurrence(self):
        cases = [
            {"use_case": "First", "test_case": "Test", "steps": [], "priority": "high"},
            {"use_case": "First", "test_case": "Test", "steps": [], "priority": "low"},
        ]
        result = generator._deduplicate_test_cases(cases)
        assert len(result) == 1
        assert result[0]["priority"] == "high"

    def test_preserves_different_cases(self):
        cases = [
            {"use_case": "Login", "test_case": "Valid Login", "steps": ["Step A"]},
            {"use_case": "Login", "test_case": "Invalid Login", "steps": ["Step B"]},
            {"use_case": "Logout", "test_case": "Valid Logout", "steps": ["Step C"]},
        ]
        result = generator._deduplicate_test_cases(cases)
        assert len(result) == 3

    def test_handles_empty_list(self):
        result = generator._deduplicate_test_cases([])
        assert result == []

    def test_handles_missing_fields(self):
        cases = [
            {"use_case": "Test"},
            {"use_case": "Test"},
        ]
        result = generator._deduplicate_test_cases(cases)
        assert len(result) == 1


class TestCleanTestCaseTitles:
    """Tests for _clean_test_case_titles function."""

    def test_removes_tc_prefix(self):
        cases = [{"test_case": "TC-001: Valid Login"}]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "Valid Login"

    def test_removes_case_prefix(self):
        cases = [{"test_case": "Case 1 - User Login"}]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "User Login"

    def test_removes_test_prefix(self):
        cases = [{"test_case": "Test #42: Something"}]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "Something"

    def test_removes_scenario_prefix(self):
        cases = [{"test_case": "Scenario 5: Happy Path"}]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "Happy Path"

    def test_removes_id_prefix(self):
        cases = [{"test_case": "ID 123 - Test Name"}]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "Test Name"

    def test_preserves_non_prefixed_titles(self):
        cases = [{"test_case": "Normal Title Without Prefix"}]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "Normal Title Without Prefix"

    def test_handles_multiple_cases(self):
        cases = [
            {"test_case": "TC-001: First"},
            {"test_case": "Case 2: Second"},
            {"test_case": "Third"},
        ]
        result = generator._clean_test_case_titles(cases)
        assert result[0]["test_case"] == "First"
        assert result[1]["test_case"] == "Second"
        assert result[2]["test_case"] == "Third"
