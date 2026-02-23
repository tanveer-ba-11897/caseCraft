
import os
import re
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader

# Define the path to templates
# Assuming templates are in "prompts/templates" relative to the project root
# or a specific configured path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "prompts", "templates")

# Initialize Jinja2 Environment
try:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
except Exception as e:
    # Fallback or error logging if needed, essentially hard to recov if templates missing
    print(f"Error loading templates from {TEMPLATES_DIR}: {e}")
    env = None

def render_template(template_name: str, **kwargs) -> str:
    """Helper to render a template safely."""
    if not env:
        raise RuntimeError(f"Template environment not initialized. Checked {TEMPLATES_DIR}")
    try:
        template = env.get_template(template_name)
        return template.render(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to render template {template_name}: {e}")


def _sanitize_input(text: str, max_length: int = 500000) -> str:
    """Sanitize user-provided text before template rendering to mitigate prompt injection."""
    if not text:
        return ""
    # Strip control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Enforce max length to prevent memory exhaustion
    if len(text) > max_length:
        text = text[:max_length] + "\n[Content truncated]"
    return text


def build_generation_prompt(
    feature_chunks: List[str], 
    product_context: str,
    app_type: str = "web"
) -> str:
    """
    Constructs the main prompt for test case generation.
    """
    joined_feature_text = _sanitize_input("\n\n".join(feature_chunks))
    product_context = _sanitize_input(product_context)

    context_section = ""
    if product_context:
        context_section = f"""
Product context (from existing product documentation):
{product_context}

Use this context to:
- Identify cross feature interactions
- Cover integration scenarios
- Avoid contradicting existing behavior
"""

    return render_template(
        "generation.j2",
        feature_text=joined_feature_text,
        context_section=context_section,
        app_type=app_type
    )

def build_cross_reference_prompt(existing_tests_json: str, checklist_text: str) -> str:
    """
    Prompt for reviewing an existing test suite against a required checklist
    and generating only the missing scenarios.
    """
    return render_template(
        "checklist_cross_reference.j2", 
        existing_tests_json=_sanitize_input(existing_tests_json), 
        checklist_text=_sanitize_input(checklist_text)
    )

def build_condensation_prompt(chunk: str) -> str:
    """
    Prompt for condensing document chunks into test-relevant bullet points.
    """
    return render_template("condensation.j2", chunk=_sanitize_input(chunk))

def build_reviewer_prompt(current_json: str) -> str:
    """
    Prompt for reviewing and improving test suite quality.
    """
    return render_template("reviewer.j2", current_json=_sanitize_input(current_json))
