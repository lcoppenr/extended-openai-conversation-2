"""Text-to-speech support for Extended OpenAI Conversation."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
import io
import logging
import re
import struct
from typing import TYPE_CHECKING, Any
import wave

from openai import OpenAIError
from propcache.api import cached_property

from homeassistant.components import tts
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_SPEED,
    DEFAULT_TTS_VOICE,
    SPEECH_LANGUAGES,
)
from .entity import ExtendedOpenAIBaseLLMEntity

if TYPE_CHECKING:
    from . import ExtendedOpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0

# OpenAI's voices, offered when the server has no voice list endpoint.
OPENAI_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]

# Where a sentence ends: after . ! ? : ; followed by whitespace, or a line break.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?:;])\s+|\n+")

# WAV format tag for uncompressed PCM.
_WAVE_FORMAT_PCM = 1


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up text-to-speech entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "tts":
            continue

        async_add_entities(
            [ExtendedOpenAITTSEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


async def split_sentences(text_stream: AsyncIterable[str]) -> AsyncGenerator[str]:
    """Yield whole sentences as they complete in a stream of text chunks."""
    buffer = ""
    async for chunk in text_stream:
        buffer += chunk
        *sentences, buffer = _SENTENCE_BREAK.split(buffer)
        for sentence in sentences:
            if sentence.strip():
                yield sentence.strip()
    if buffer.strip():
        yield buffer.strip()


def split_wav(data: bytes) -> tuple[tuple[int, int, int], bytes]:
    """Return ((rate, sample width, channels), PCM frames) from WAV bytes.

    Everything after the data chunk header is taken as PCM, so streamed WAV
    whose size fields are unset (0 or 0xFFFFFFFF) is read correctly.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HomeAssistantError("Text-to-speech server did not return WAV audio")

    audio_format: tuple[int, int, int] | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (chunk_size,) = struct.unpack_from("<I", data, offset + 4)
        body = offset + 8
        if chunk_id == b"fmt ":
            format_tag, channels, rate = struct.unpack_from("<HHI", data, body)
            (bits,) = struct.unpack_from("<H", data, body + 14)
            if format_tag != _WAVE_FORMAT_PCM:
                raise HomeAssistantError(
                    f"Unsupported WAV encoding from text-to-speech server: {format_tag}"
                )
            audio_format = (rate, bits // 8, channels)
        elif chunk_id == b"data":
            if audio_format is None:
                break
            return audio_format, data[body:]
        offset = body + chunk_size + (chunk_size & 1)

    raise HomeAssistantError("Text-to-speech server returned an incomplete WAV file")


def wav_header(rate: int, sample_width: int, channels: int) -> bytes:
    """Return a WAV header with a frame count of 0, meaning a stream."""
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setframerate(rate)
            wav_file.setsampwidth(sample_width)
            wav_file.setnchannels(channels)
        return wav_io.getvalue()


class ExtendedOpenAITTSEntity(tts.TextToSpeechEntity, ExtendedOpenAIBaseLLMEntity):
    """Text-to-speech entity using an OpenAI-compatible speech endpoint.

    Speech is synthesized one sentence at a time, so it starts playing
    while the conversation agent is still writing the rest of its answer.
    """

    entry: ExtendedOpenAIConfigEntry

    _attr_supported_languages = SPEECH_LANGUAGES
    # Unused, but required by the base class; the model follows the text.
    _attr_default_language = "en-US"

    def __init__(
        self, entry: ExtendedOpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._attr_supported_options = [tts.ATTR_VOICE]
        self._voices: list[tts.Voice] = [
            tts.Voice(voice, voice) for voice in OPENAI_VOICES
        ]

    @cached_property
    def default_options(self) -> dict[str, Any]:
        """Return the default options."""
        return {
            tts.ATTR_VOICE: self.subentry.data.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE)
        }

    async def async_added_to_hass(self) -> None:
        """Load the server's voice list, if it has one."""
        await super().async_added_to_hass()
        model = self.subentry.data.get(CONF_CHAT_MODEL, DEFAULT_TTS_MODEL)
        try:
            response = await self._client.get(
                "/audio/voices", cast_to=object, options={"params": {"model": model}}
            )
        except OpenAIError as err:
            _LOGGER.debug("No voice list from server, offering OpenAI voices: %s", err)
            return

        voices = response.get("voices") if isinstance(response, dict) else None
        if isinstance(voices, list) and voices:
            self._voices = [tts.Voice(str(voice), str(voice)) for voice in voices]

    @callback
    def async_get_supported_voices(self, language: str) -> list[tts.Voice]:
        """Return the voices the model offers."""
        return self._voices

    async def _async_synthesize(self, text: str, options: dict[str, Any]) -> bytes:
        """Return WAV audio for text."""
        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.subentry.data.get(CONF_CHAT_MODEL, DEFAULT_TTS_MODEL),
                voice=options[tts.ATTR_VOICE],
                input=text,
                speed=self.subentry.data.get(CONF_TTS_SPEED, DEFAULT_TTS_SPEED),
                response_format="wav",
            ) as response:
                return await response.read()
        except OpenAIError as err:
            _LOGGER.exception("Error during text-to-speech")
            raise HomeAssistantError(err) from err

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> tts.TtsAudioType:
        """Return WAV audio for a whole message."""
        return "wav", await self._async_synthesize(
            message, {**self.default_options, **options}
        )

    async def async_stream_tts_audio(
        self, request: tts.TTSAudioRequest
    ) -> tts.TTSAudioResponse:
        """Stream WAV audio, synthesizing each sentence as it arrives."""
        options = {**self.default_options, **request.options}

        async def data_gen() -> AsyncGenerator[bytes]:
            stream_format: tuple[int, int, int] | None = None
            async for sentence in split_sentences(request.message_gen):
                audio_format, pcm = split_wav(
                    await self._async_synthesize(sentence, options)
                )
                if stream_format is None:
                    stream_format = audio_format
                    yield wav_header(*audio_format)
                elif audio_format != stream_format:
                    raise HomeAssistantError(
                        "Text-to-speech server changed audio format mid-stream"
                    )
                yield pcm

        return tts.TTSAudioResponse("wav", data_gen())
