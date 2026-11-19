"""Tests for tool-call results across HA 2026.9 and 2026.10+ ToolResultContent."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.extended_openai_conversation.entity import (
    ExtendedOpenAIBaseLLMEntity,
    _convert_content_to_param,
    _make_tool_result_content,
)
from homeassistant.components import conversation
from homeassistant.helpers import llm


def test_tool_result_content_round_trips_to_openai_message(caplog):
    """A tool result is built for this HA version and read back without warnings."""
    content = _make_tool_result_content(
        agent_id="conversation.test",
        tool_call_id="call_1",
        tool_name="turn_on_light",
        data={"result": "ok"},
    )

    with caplog.at_level(logging.WARNING):
        messages = _convert_content_to_param([content])

    assert messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result":"ok"}'}
    ]
    assert "deprecated" not in caplog.text


@pytest.mark.parametrize("delay", [None, {"seconds": 5}])
async def test_execute_function_tool_returns_readable_result(hass, delay):
    """_execute_function_tool (device control path) produces a usable result."""
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    entity.entry = MagicMock()
    # Background (delayed) calls are scheduled, not awaited, in this test
    entity.entry.async_create_task = MagicMock(
        side_effect=lambda _hass, coro: coro.close()
    )

    function = MagicMock()
    function.execute = AsyncMock(return_value="done")
    args = {"entity_id": "light.foyer"}
    if delay is not None:
        args["delay"] = delay
    tool_input = llm.ToolInput(
        id="call_1", tool_name="execute_services", tool_args=args
    )

    with patch(
        "custom_components.extended_openai_conversation.entity.get_function",
        return_value=function,
    ):
        content = await entity._execute_function_tool(
            {"function": {"type": "native"}}, tool_input, None, []
        )

    assert isinstance(content, conversation.ToolResultContent)
    assert content.tool_call_id == "call_1"
    message = _convert_content_to_param([content])[0]
    expected = "Scheduled" if delay is not None else "done"
    assert json.loads(message["content"]) == {"result": expected}
