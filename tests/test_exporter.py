"""
Tests for core.exporter module.
Tests: path validation, Excel export, JSON export.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.schema import TestSuite, TestCase
from core import exporter


@pytest.fixture
def sample_test_suite():
    """Create a sample TestSuite for testing."""
    return TestSuite(
        feature_name="Test Feature",
        source_document="test_feature_doc.txt",
        test_cases=[
            TestCase(
                use_case="User Login",
                test_case="Valid credentials login",
                test_type="functionality",
                preconditions=["User account exists", "User is on login page"],
                test_data={"username": "testuser", "password": "testpass"},
                steps=["Enter username", "Enter password", "Click login"],
                priority="high",
                tags=["login", "auth"],
                expected_results=["User is redirected to dashboard"],
                actual_results=[],
                dependencies=["Database connection"],
            ),
            TestCase(
                use_case="User Logout",
                test_case="Successful logout",
                test_type="functionality",
                preconditions=["User is logged in"],
                test_data={},
                steps=["Click logout button"],
                priority="medium",
                tags=["logout"],
                expected_results=["User is redirected to login page"],
                actual_results=[],
                dependencies=[],
            ),
        ],
    )


class TestValidateOutputPath:
    """Tests for _validate_output_path function."""

    def test_rejects_path_traversal(self):
        """Should reject paths that escape output directory."""
        with pytest.raises(PermissionError):
            exporter._validate_output_path("../../../etc/passwd")

    def test_rejects_absolute_path_outside_output_dir(self):
        """Should reject absolute paths outside configured output directory."""
        with pytest.raises(PermissionError):
            exporter._validate_output_path("/tmp/malicious_file.xlsx")

    def test_accepts_relative_path_in_output_dir(self):
        """Should accept paths within output directory."""
        # This should not raise - the actual output_dir is 'outputs'
        result = exporter._validate_output_path("outputs/test_output.xlsx")
        assert "outputs" in result
        assert "test_output.xlsx" in result

    def test_accepts_nested_path_in_output_dir(self):
        """Should accept nested paths within output directory."""
        result = exporter._validate_output_path("outputs/subdir/test.xlsx")
        assert "outputs" in result
        assert "subdir" in result


class TestJoinLines:
    """Tests for _join_lines helper function."""

    def test_joins_multiple_items(self):
        result = exporter._join_lines(["line1", "line2", "line3"])
        assert result == "line1\nline2\nline3"

    def test_handles_single_item(self):
        result = exporter._join_lines(["only one"])
        assert result == "only one"

    def test_handles_empty_list(self):
        result = exporter._join_lines([])
        assert result == ""


class TestExportJson:
    """Tests for JSON export functionality."""

    def test_exports_valid_json(self, sample_test_suite, tmp_path):
        """Exported JSON should be valid and parseable."""
        with patch.object(exporter, '_validate_output_path') as mock_validate:
            output_path = tmp_path / "test_output.json"
            mock_validate.return_value = str(output_path)
            
            exporter._export_json(sample_test_suite, str(output_path))
            
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert data["feature_name"] == "Test Feature"
            assert len(data["test_cases"]) == 2

    def test_json_contains_all_test_case_fields(self, sample_test_suite, tmp_path):
        """Exported JSON should contain all TestCase fields."""
        with patch.object(exporter, '_validate_output_path') as mock_validate:
            output_path = tmp_path / "test_output.json"
            mock_validate.return_value = str(output_path)
            
            exporter._export_json(sample_test_suite, str(output_path))
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            test_case = data["test_cases"][0]
            assert "use_case" in test_case
            assert "test_case" in test_case
            assert "preconditions" in test_case
            assert "test_data" in test_case
            assert "steps" in test_case
            assert "priority" in test_case
            assert "expected_results" in test_case


class TestExport:
    """Tests for the main export function."""

    def test_routes_to_excel_export(self, sample_test_suite):
        """Should route 'excel' format to _export_excel."""
        with patch.object(exporter, '_export_excel') as mock_excel:
            with patch.object(exporter, '_validate_output_path', return_value="/tmp/out.xlsx"):
                exporter.export(sample_test_suite, "excel", "outputs/test.xlsx")
                mock_excel.assert_called_once()

    def test_routes_to_json_export(self, sample_test_suite):
        """Should route 'json' format to _export_json."""
        with patch.object(exporter, '_export_json') as mock_json:
            with patch.object(exporter, '_validate_output_path', return_value="/tmp/out.json"):
                exporter.export(sample_test_suite, "json", "outputs/test.json")
                mock_json.assert_called_once()

    def test_handles_uppercase_format(self, sample_test_suite):
        """Should handle format string case-insensitively."""
        with patch.object(exporter, '_export_excel') as mock_excel:
            with patch.object(exporter, '_validate_output_path', return_value="/tmp/out.xlsx"):
                exporter.export(sample_test_suite, "EXCEL", "outputs/test.xlsx")
                mock_excel.assert_called_once()

    def test_handles_whitespace_in_format(self, sample_test_suite):
        """Should handle format string with whitespace."""
        with patch.object(exporter, '_export_json') as mock_json:
            with patch.object(exporter, '_validate_output_path', return_value="/tmp/out.json"):
                exporter.export(sample_test_suite, "  json  ", "outputs/test.json")
                mock_json.assert_called_once()

    def test_raises_for_unsupported_format(self, sample_test_suite):
        """Should raise ValueError for unsupported formats."""
        with pytest.raises(ValueError) as excinfo:
            exporter.export(sample_test_suite, "xml", "outputs/test.xml")
        
        assert "Unsupported output format" in str(excinfo.value)
