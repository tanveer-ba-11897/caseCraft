"""Tests for unescaped-quote repair and json_repair fallback."""
import json
import pytest
from core.generator import _clean_json_output, _fix_unescaped_inner_quotes, _repair_json


class TestFixUnescapedInnerQuotes:
    def test_escapes_inner_quotes(self):
        # "Click the "Save" button" should become "Click the \"Save\" button"
        bad = '{"steps": ["Click the "Save" button"]}'
        fixed = _fix_unescaped_inner_quotes(bad)
        parsed = json.loads(fixed)
        assert parsed["steps"][0] == 'Click the "Save" button'

    def test_multiple_inner_quotes(self):
        bad = '{"a": "Go to "Leads" module", "b": "Click "Map View" tab"}'
        fixed = _fix_unescaped_inner_quotes(bad)
        parsed = json.loads(fixed)
        assert parsed["a"] == 'Go to "Leads" module'
        assert parsed["b"] == 'Click "Map View" tab'

    def test_valid_json_unchanged(self):
        good = '{"steps": ["Click the button", "Verify the message"]}'
        fixed = _fix_unescaped_inner_quotes(good)
        assert json.loads(fixed) == json.loads(good)

    def test_already_escaped_quotes_untouched(self):
        good = '{"steps": ["Click the \\"Save\\" button"]}'
        fixed = _fix_unescaped_inner_quotes(good)
        parsed = json.loads(fixed)
        assert parsed["steps"][0] == 'Click the "Save" button'

    def test_nested_arrays_and_objects(self):
        good = '{"a": [{"b": "value"}, {"c": "other"}]}'
        fixed = _fix_unescaped_inner_quotes(good)
        assert json.loads(fixed) == json.loads(good)

    def test_empty_strings(self):
        good = '{"a": "", "b": []}'
        fixed = _fix_unescaped_inner_quotes(good)
        assert json.loads(fixed) == json.loads(good)


class TestRepairJsonInnerQuotes:
    def test_full_pipeline(self):
        bad = '{"test_cases": [{"test_case": "Map view hidden", "steps": ["Navigate to "Leads" module"]}]}'
        fixed = _repair_json(bad)
        parsed = json.loads(fixed)
        assert parsed["test_cases"][0]["steps"][0] == 'Navigate to "Leads" module'


class TestCleanJsonOutputWithInnerQuotes:
    def test_code_block_with_inner_quotes(self):
        bad = '```json\n{"test_cases": [{"steps": ["Click "Save" button"]}]}\n```'
        cleaned = _clean_json_output(bad)
        parsed = json.loads(cleaned)
        assert parsed["test_cases"][0]["steps"][0] == 'Click "Save" button'


class TestJsonRepairFallback:
    def test_library_handles_trailing_comma(self):
        from json_repair import repair_json
        bad = '{"a": [1, 2, 3,]}'
        repaired = repair_json(bad, return_objects=False)
        parsed = json.loads(repaired)
        assert parsed == {"a": [1, 2, 3]}

    def test_library_handles_inner_quotes(self):
        from json_repair import repair_json
        bad = '{"a": "value with "quotes" inside"}'
        repaired = repair_json(bad, return_objects=False)
        parsed = json.loads(repaired)
        assert "quotes" in parsed["a"]
