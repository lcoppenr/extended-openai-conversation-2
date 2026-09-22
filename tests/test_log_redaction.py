"""Tests for prompt-log redaction of inline image data."""

from custom_components.extended_openai_conversation.helpers import redact_image_data


def test_redact_image_data_replaces_data_urls_only():
    """Prompt logging strips inline image bytes but keeps everything else."""
    messages = [
        {"role": "system", "content": "hi"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + "A" * 5000},
                },
                {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}},
            ],
        },
    ]

    result = redact_image_data(messages)

    parts = result[1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,<")
    assert "AAAA" not in parts[1]["image_url"]["url"]
    assert parts[2]["image_url"]["url"] == "https://x/y.jpg"
    # Input is not mutated
    assert messages[1]["content"][1]["image_url"]["url"].endswith("A" * 10)
