"""Speech-to-text support for Extended OpenAI Conversation."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
from typing import TYPE_CHECKING
import wave

from openai import OpenAIError

from homeassistant.components import stt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_CHAT_MODEL, DEFAULT_STT_MODEL, SPEECH_LANGUAGES
from .entity import ExtendedOpenAIBaseLLMEntity

if TYPE_CHECKING:
    from . import ExtendedOpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up speech-to-text entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [ExtendedOpenAISTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ExtendedOpenAISTTEntity(stt.SpeechToTextEntity, ExtendedOpenAIBaseLLMEntity):
    """Speech-to-text entity using an OpenAI-compatible transcriptions endpoint."""

    entry: ExtendedOpenAIConfigEntry

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SPEECH_LANGUAGES

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Send the collected audio to the transcriptions endpoint."""
        audio = bytearray()
        async for chunk in stream:
            audio.extend(chunk)

        # The stream is raw PCM; the endpoint needs a WAV file.
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(metadata.channel.value)
            wav_file.setsampwidth(metadata.bit_rate.value // 8)
            wav_file.setframerate(metadata.sample_rate.value)
            wav_file.writeframes(bytes(audio))

        try:
            response = await self._client.audio.transcriptions.create(
                model=self.subentry.data.get(CONF_CHAT_MODEL, DEFAULT_STT_MODEL),
                file=("audio.wav", wav_buffer.getvalue()),
                response_format="json",
                language=metadata.language.split("-")[0],
            )
        except OpenAIError:
            _LOGGER.exception("Error during speech-to-text")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        text = response.text.strip()
        if not text:
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
