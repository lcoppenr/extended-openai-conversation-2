"""Tests for ai_task structured output schema conversion."""

import pytest

from custom_components.extended_openai_conversation.entity import (
    _format_structured_output,
)

try:  # HA 2026.10+
    from homeassistant.components.ai_task.services import _validate_structure_fields
except ImportError:  # HA 2026.9
    from homeassistant.components.ai_task import _validate_structure_fields


def test_text_field_structure_converts():
    """The minimal ai_task structure (one text field) converts to a strict object.

    Regression: HA 2026.9's selector serializer returns probatio's UNSUPPORTED
    sentinel, which voluptuous_openapi.convert() passed through as the schema.
    """
    schema = _validate_structure_fields({"color": {"selector": {"text": {}}}})

    result = _format_structured_output(schema, None)

    assert result["type"] == "object"
    assert result["additionalProperties"] is False
    assert result["required"] == ["color"]
    # Optional field becomes required-but-nullable for OpenAI strict mode
    assert "null" in result["properties"]["color"]["type"]


def test_mixed_selector_structure_converts():
    """Boolean, number, select and multi-select fields all convert."""
    schema = _validate_structure_fields(
        {
            "person": {"selector": {"boolean": {}}, "required": True},
            "count": {"selector": {"number": {"min": 0, "max": 20}}},
            "kind": {
                "selector": {"select": {"options": ["person", "car", "none"]}},
                "required": True,
            },
            "tags": {"selector": {"select": {"options": ["a", "b"], "multiple": True}}},
        }
    )

    result = _format_structured_output(schema, None)

    props = result["properties"]
    assert set(result["required"]) == {"person", "count", "kind", "tags"}
    assert props["person"]["type"] == "boolean"
    assert props["kind"]["enum"] == ["person", "car", "none"]
    assert "null" in props["count"]["type"]
    assert props["tags"]["items"]["enum"] == ["a", "b"]


def test_unrepresentable_structure_raises_clear_error():
    """A free-form object field can't be expressed in strict mode and says so."""
    schema = _validate_structure_fields({"obj": {"selector": {"object": {}}}})

    with pytest.raises(Exception, match="OpenAI"):
        _format_structured_output(schema, None)
