from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator


# Valid values for test_type and priority
TestType = Literal[
    "functionality", "ui", "performance", "integration",
    "usability", "database", "security", "acceptance"
]
Priority = Literal["high", "medium", "low"]


class TestCase(BaseModel):
    """
    Represents a single test case aligned with schema.md.
    """

    use_case: str = Field(
        ...,
        description="High-level use case this test belongs to"
    )

    test_case: str = Field(
        ...,
        description="Short descriptive name of the test case"
    )

    test_type: TestType = Field(
        default="functionality",
        description="Type of test case: functionality, ui, performance, integration, usability, database, security, or acceptance"
    )

    preconditions: List[str] = Field(
        default_factory=list,
        description="Conditions that must be satisfied before execution"
    )

    test_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data required for the test (allows nested JSON values)"
    )

    steps: List[str] = Field(
        ...,
        description="Step-by-step actions to execute the test"
    )

    priority: Priority = Field(
        ...,
        description="Priority of the test case: high, medium, or low"
    )

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, v: str) -> str:
        """Normalize priority to lowercase. Defaults to 'medium' if invalid."""
        if not isinstance(v, str):
            return "medium"
        normalized = v.strip().lower()
        if normalized in ("high", "medium", "low"):
            return normalized
        return "medium"

    @field_validator("test_type", mode="before")
    @classmethod
    def normalize_test_type(cls, v: str) -> str:
        """Normalize test_type to lowercase. Defaults to 'functionality' if invalid."""
        valid_types = {
            "functionality", "ui", "performance", "integration",
            "usability", "database", "security", "acceptance"
        }
        if not isinstance(v, str):
            return "functionality"
        normalized = v.strip().lower()
        if normalized in valid_types:
            return normalized
        return "functionality"

    tags: List[str] = Field(
        default_factory=list,
        description="Labels for categorization or filtering"
    )

    dependencies: List[str] = Field(
        default_factory=list,
        description="List of other test cases or conditions this test depends on"
    )

    expected_results: List[str] = Field(
        ...,
        description="Expected outcome after executing the steps"
    )

    actual_results: List[str] = Field(
        default_factory=list,
        description="Actual results observed during execution"
    )


class TestSuite(BaseModel):
    """
    Collection of test cases derived from a feature or document.
    """

    feature_name: str = Field(
        ...,
        description="Name of the feature under test"
    )

    source_document: str = Field(
        ...,
        description="Document used to generate the test cases"
    )

    test_cases: List[TestCase] = Field(
        ...,
        description="Generated test cases"
    )
