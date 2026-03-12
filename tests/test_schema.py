"""
Tests for core.schema module.
Tests: TestCase validation, TestSuite validation, field validators.
"""

import pytest
from core.schema import TestCase, TestSuite


@pytest.fixture
def minimal_test_case_data():
    """Minimal required data for a TestCase."""
    return {
        "use_case": "Test",
        "test_case": "Test",
        "steps": ["Step 1"],
        "priority": "medium",
        "expected_results": ["Expected result"],
    }


class TestTestCasePriorityValidation:
    """Tests for priority field validation."""

    def test_accepts_valid_priority_values(self, minimal_test_case_data):
        """Should accept high, medium, low priorities."""
        for priority in ["high", "medium", "low"]:
            data = {**minimal_test_case_data, "priority": priority}
            tc = TestCase(**data)
            assert tc.priority == priority

    def test_normalizes_uppercase_priority(self, minimal_test_case_data):
        """Should normalize uppercase priorities to lowercase."""
        data = {**minimal_test_case_data, "priority": "HIGH"}
        tc = TestCase(**data)
        assert tc.priority == "high"

    def test_normalizes_mixed_case_priority(self, minimal_test_case_data):
        """Should normalize mixed case priorities."""
        data = {**minimal_test_case_data, "priority": "MeDiUm"}
        tc = TestCase(**data)
        assert tc.priority == "medium"

    def test_defaults_invalid_priority_to_medium(self, minimal_test_case_data):
        """Should default invalid priority to medium."""
        data = {**minimal_test_case_data, "priority": "invalid_priority"}
        tc = TestCase(**data)
        assert tc.priority == "medium"

    def test_handles_empty_priority(self, minimal_test_case_data):
        """Should handle empty priority string."""
        data = {**minimal_test_case_data, "priority": ""}
        tc = TestCase(**data)
        assert tc.priority == "medium"


class TestTestCaseTestTypeValidation:
    """Tests for test_type field validation."""

    def test_accepts_valid_test_types(self, minimal_test_case_data):
        """Should accept valid test types."""
        valid_types = [
            "functionality", "ui", "integration", "performance",
            "security", "usability", "database", "acceptance",
        ]
        for test_type in valid_types:
            data = {**minimal_test_case_data, "test_type": test_type}
            tc = TestCase(**data)
            assert tc.test_type == test_type

    def test_normalizes_uppercase_test_type(self, minimal_test_case_data):
        """Should normalize uppercase test types to lowercase."""
        data = {**minimal_test_case_data, "test_type": "FUNCTIONALITY"}
        tc = TestCase(**data)
        assert tc.test_type == "functionality"

    def test_defaults_invalid_test_type_to_functionality(self, minimal_test_case_data):
        """Should default invalid test_type to functionality."""
        data = {**minimal_test_case_data, "test_type": "invalid_type"}
        tc = TestCase(**data)
        assert tc.test_type == "functionality"

    def test_handles_empty_test_type(self, minimal_test_case_data):
        """Should handle empty test_type string."""
        data = {**minimal_test_case_data, "test_type": ""}
        tc = TestCase(**data)
        assert tc.test_type == "functionality"


class TestTestCaseDefaultValues:
    """Tests for TestCase default values."""

    def test_default_preconditions_is_empty_list(self, minimal_test_case_data):
        tc = TestCase(**minimal_test_case_data)
        assert tc.preconditions == []

    def test_default_test_data_is_empty_dict(self, minimal_test_case_data):
        tc = TestCase(**minimal_test_case_data)
        assert tc.test_data == {}

    def test_default_tags_is_empty_list(self, minimal_test_case_data):
        tc = TestCase(**minimal_test_case_data)
        assert tc.tags == []

    def test_default_actual_results_is_empty_list(self, minimal_test_case_data):
        tc = TestCase(**minimal_test_case_data)
        assert tc.actual_results == []

    def test_default_dependencies_is_empty_list(self, minimal_test_case_data):
        tc = TestCase(**minimal_test_case_data)
        assert tc.dependencies == []

    def test_default_test_type_is_functionality(self, minimal_test_case_data):
        tc = TestCase(**minimal_test_case_data)
        assert tc.test_type == "functionality"


class TestTestSuite:
    """Tests for TestSuite model."""

    def test_creates_suite_with_test_cases(self, minimal_test_case_data):
        suite = TestSuite(
            feature_name="Login Feature",
            source_document="login_spec.txt",
            test_cases=[
                TestCase(**minimal_test_case_data),
                TestCase(**{**minimal_test_case_data, "test_case": "Another test"}),
            ],
        )
        assert suite.feature_name == "Login Feature"
        assert suite.source_document == "login_spec.txt"
        assert len(suite.test_cases) == 2

    def test_can_serialize_to_json(self, minimal_test_case_data):
        suite = TestSuite(
            feature_name="Test",
            source_document="test.txt",
            test_cases=[TestCase(**minimal_test_case_data)],
        )
        json_str = suite.model_dump_json()
        assert "Test" in json_str
        assert "test_cases" in json_str
