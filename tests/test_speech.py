"""Tests for the speech-to-text and text-to-speech platforms."""

from collections.abc import AsyncGenerator
import io
import struct
from unittest.mock import AsyncMock, MagicMock
import wave

from openai import OpenAIError
import pytest

from custom_components.extended_openai_conversation.config_flow import (
    ExtendedOpenAIConversationConfigFlow,
    ExtendedOpenAISTTSubentryFlowHandler,
    ExtendedOpenAITTSSubentryFlowHandler,
)
from custom_components.extended_openai_conversation.stt import (
    ExtendedOpenAISTTEntity,
    async_setup_entry as stt_setup_entry,
)
from custom_components.extended_openai_conversation.tts import (
    OPENAI_VOICES,
    ExtendedOpenAITTSEntity,
    async_setup_entry as tts_setup_entry,
    split_sentences,
    split_wav,
    wav_header,
)
from homeassistant.components import stt, tts
from homeassistant.exceptions import HomeAssistantError


def _subentry(subentry_type: str, subentry_id: str, data: dict | None = None):
    return MagicMock(
        subentry_type=subentry_type,
        subentry_id=subentry_id,
        title=f"{subentry_type} entry",
        data=data or {},
    )


def _wav(pcm: bytes, rate: int = 24000, width: int = 2, channels: int = 1) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setframerate(rate)
            wav_file.setsampwidth(width)
            wav_file.setnchannels(channels)
            wav_file.writeframes(pcm)
        return wav_io.getvalue()


async def _chunks(*parts: str) -> AsyncGenerator[str]:
    for part in parts:
        yield part


def _entry(client: MagicMock, *subentries: MagicMock) -> MagicMock:
    entry = MagicMock()
    entry.runtime_data = client
    entry.subentries = {s.subentry_id: s for s in subentries}
    return entry


# --- setup -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup_entry", "subentry_type", "entity_class"),
    [
        (stt_setup_entry, "stt", ExtendedOpenAISTTEntity),
        (tts_setup_entry, "tts", ExtendedOpenAITTSEntity),
    ],
)
async def test_setup_adds_entity_only_for_matching_subentry(
    hass, setup_entry, subentry_type, entity_class
):
    """Each speech platform adds an entity for its own subentry type only."""
    entry = _entry(
        MagicMock(),
        _subentry(subentry_type, "sub_speech"),
        _subentry("conversation", "sub_conv"),
    )
    async_add_entities = MagicMock()

    await setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_called_once()
    (entities,), kwargs = async_add_entities.call_args
    assert isinstance(entities[0], entity_class)
    assert kwargs["config_subentry_id"] == "sub_speech"


def test_config_flow_offers_speech_subentries():
    """The stt and tts subentry types are registered."""
    types = ExtendedOpenAIConversationConfigFlow.async_get_supported_subentry_types(
        MagicMock()
    )
    assert types["stt"] is ExtendedOpenAISTTSubentryFlowHandler
    assert types["tts"] is ExtendedOpenAITTSSubentryFlowHandler


# --- speech-to-text --------------------------------------------------------


def _metadata() -> stt.SpeechMetadata:
    return stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )


async def _audio() -> AsyncGenerator[bytes]:
    yield b"\x01\x00" * 100
    yield b"\x02\x00" * 100


def _stt_entity(client: MagicMock) -> ExtendedOpenAISTTEntity:
    subentry = _subentry("stt", "sub_stt", {"chat_model": "stt"})
    return ExtendedOpenAISTTEntity(_entry(client, subentry), subentry)


async def test_stt_sends_wav_and_returns_text():
    """Streamed PCM is wrapped in a WAV file and the transcript returned."""
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text=" Turn off the lights. ")
    )

    result = await _stt_entity(client).async_process_audio_stream(_metadata(), _audio())

    assert result.result == stt.SpeechResultState.SUCCESS
    assert result.text == "Turn off the lights."
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "stt"
    assert kwargs["language"] == "en"
    name, data = kwargs["file"]
    assert name == "audio.wav"
    with wave.open(io.BytesIO(data)) as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 200


@pytest.mark.parametrize(
    "create",
    [
        AsyncMock(side_effect=OpenAIError("down")),
        AsyncMock(return_value=MagicMock(text="  ")),
    ],
)
async def test_stt_error_on_failure_or_empty_transcript(create):
    """A server error or an empty transcript is reported as an error result."""
    client = MagicMock()
    client.audio.transcriptions.create = create

    result = await _stt_entity(client).async_process_audio_stream(_metadata(), _audio())

    assert result.result == stt.SpeechResultState.ERROR
    assert result.text is None


# --- text-to-speech helpers ------------------------------------------------


async def test_split_sentences_across_chunks():
    """Sentences are yielded once complete, even when split across chunks."""
    sentences = [
        s
        async for s in split_sentences(
            _chunks("The lights are ", "off. The door", " is locked!\nDone")
        )
    ]
    assert sentences == ["The lights are off.", "The door is locked!", "Done"]


async def test_split_sentences_keeps_decimals_together():
    """A period inside a number is not a sentence break."""
    sentences = [s async for s in split_sentences(_chunks("It is 21.5 degrees."))]
    assert sentences == ["It is 21.5 degrees."]


async def test_split_sentences_empty_message():
    """Whitespace-only text yields nothing."""
    assert [s async for s in split_sentences(_chunks("  ", "\n"))] == []


def test_split_wav_reads_format_and_frames():
    """Format and PCM come back from a normal WAV file."""
    audio_format, pcm = split_wav(_wav(b"\x01\x02" * 10))
    assert audio_format == (24000, 2, 1)
    assert pcm == b"\x01\x02" * 10


def test_split_wav_streamed_header_and_extra_chunk():
    """Unset size fields and a chunk before fmt are handled."""
    fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    data = (
        b"RIFF\xff\xff\xff\xffWAVE"
        + b"LIST"
        + struct.pack("<I", 3)
        + b"abc\x00"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data\xff\xff\xff\xff"
        + b"\x05\x06" * 4
    )
    assert split_wav(data) == ((24000, 2, 1), b"\x05\x06" * 4)


@pytest.mark.parametrize(
    "data",
    [
        b"ID3 not a wav at all",
        b"RIFF\x00\x00\x00\x00WAVE",
        b"RIFF\x00\x00\x00\x00WAVEfmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
        + b"data\x00\x00\x00\x00",
    ],
)
def test_split_wav_rejects_bad_audio(data):
    """Non-WAV, incomplete, and non-PCM audio raise."""
    with pytest.raises(HomeAssistantError):
        split_wav(data)


def test_wav_header_is_streaming_header():
    """The header describes the format with zero frames."""
    with wave.open(io.BytesIO(wav_header(24000, 2, 1))) as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 0


# --- text-to-speech entity -------------------------------------------------


def _tts_entity(client: MagicMock, data: dict | None = None):
    subentry = _subentry(
        "tts", "sub_tts", data or {"chat_model": "tts", "voice": "af_heart"}
    )
    return ExtendedOpenAITTSEntity(_entry(client, subentry), subentry)


async def test_tts_stream_one_header_then_pcm_per_sentence():
    """Each sentence is synthesized separately behind a single WAV header."""
    entity = _tts_entity(MagicMock())
    entity._async_synthesize = AsyncMock(
        side_effect=[_wav(b"\x01\x00" * 3), _wav(b"\x02\x00" * 5)]
    )

    response = await entity.async_stream_tts_audio(
        tts.TTSAudioRequest(
            language="en-US",
            options={},
            message_gen=_chunks("Lights off. ", "Door locked."),
        )
    )
    audio = b"".join([chunk async for chunk in response.data_gen])

    assert response.extension == "wav"
    assert [c.args[0] for c in entity._async_synthesize.call_args_list] == [
        "Lights off.",
        "Door locked.",
    ]
    assert entity._async_synthesize.call_args.args[1]["voice"] == "af_heart"
    header = wav_header(24000, 2, 1)
    assert audio == header + b"\x01\x00" * 3 + b"\x02\x00" * 5


async def test_tts_stream_format_change_raises():
    """A sentence in a different audio format stops the stream."""
    entity = _tts_entity(MagicMock())
    entity._async_synthesize = AsyncMock(
        side_effect=[_wav(b"\x00\x00", rate=24000), _wav(b"\x00\x00", rate=16000)]
    )

    response = await entity.async_stream_tts_audio(
        tts.TTSAudioRequest(
            language="en-US", options={}, message_gen=_chunks("One. Two.")
        )
    )
    with pytest.raises(HomeAssistantError):
        async for _ in response.data_gen:
            pass


async def test_tts_synthesize_request():
    """The speech request carries the configured model, voice and speed."""
    client = MagicMock()
    response = MagicMock()
    response.read = AsyncMock(return_value=b"RIFF")
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=response)
    stream.__aexit__ = AsyncMock(return_value=None)
    client.audio.speech.with_streaming_response.create = MagicMock(return_value=stream)
    entity = _tts_entity(
        client, {"chat_model": "tts", "voice": "af_heart", "speed": 1.1}
    )

    assert await entity.async_get_tts_audio("Hello.", "en-US", {}) == ("wav", b"RIFF")
    kwargs = client.audio.speech.with_streaming_response.create.call_args.kwargs
    assert kwargs == {
        "model": "tts",
        "voice": "af_heart",
        "input": "Hello.",
        "speed": 1.1,
        "response_format": "wav",
    }


async def test_tts_synthesize_error_raises():
    """A server error surfaces as HomeAssistantError."""
    client = MagicMock()
    client.audio.speech.with_streaming_response.create = MagicMock(
        side_effect=OpenAIError("down")
    )
    with pytest.raises(HomeAssistantError):
        await _tts_entity(client).async_get_tts_audio("Hello.", "en-US", {})


async def test_tts_loads_server_voices(hass):
    """Voices come from the server's voice list when it has one."""
    client = MagicMock()
    client.get = AsyncMock(return_value={"voices": ["af_heart", "bm_george"]})
    entity = _tts_entity(client)
    entity.hass = hass

    await entity.async_added_to_hass()

    assert [v.voice_id for v in entity.async_get_supported_voices("en-US")] == [
        "af_heart",
        "bm_george",
    ]
    assert client.get.call_args.kwargs["options"] == {"params": {"model": "tts"}}


async def test_tts_falls_back_to_openai_voices(hass):
    """Without a voice list endpoint, OpenAI's voices are offered."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=OpenAIError("404"))
    entity = _tts_entity(client)
    entity.hass = hass

    await entity.async_added_to_hass()

    assert [
        v.voice_id for v in entity.async_get_supported_voices("en-US")
    ] == OPENAI_VOICES
