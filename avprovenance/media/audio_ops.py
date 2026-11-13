"""PCM audio operations with explicit clipping and length rules."""

from __future__ import annotations

import math
from typing import Any

from .errors import MediaOperationFailure
from .wav import AudioClip, decode_wav, encode_wav, replace_channels


PCM_MIN = -32768
PCM_MAX = 32767


def _clip_sample(value: float) -> int:
    if value >= PCM_MAX:
        return PCM_MAX
    if value <= PCM_MIN:
        return PCM_MIN
    return int(round(value))


def _gain_factor(db: float) -> float:
    return math.pow(10.0, float(db) / 20.0)


def trim_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    start = int(round(float(operation["start_seconds"]) * clip.sample_rate))
    duration = int(round(float(operation["duration_seconds"]) * clip.sample_rate))
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "AUDIO_TRIM_OUT_OF_RANGE",
            "The requested audio trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, duration))
    if end <= start:
        raise MediaOperationFailure(
            "AUDIO_TRIM_EMPTY",
            "The requested audio trim produces no sample frames.",
        )
    # Preserve where the surviving head lives on the original source timeline.
    new_origin = clip.source_start_frame + start
    return replace_channels(
        clip,
        (values[start:end] for values in clip.channels),
        source_start_frame=new_origin,
    )


def gain_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    factor = _gain_factor(float(operation["db"]))
    return replace_channels(
        clip,
        (
            [_clip_sample(value * factor) for value in values]
            for values in clip.channels
        ),
    )


def normalize_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    peak = max(
        (abs(value) for channel in clip.channels for value in channel),
        default=0,
    )
    if peak == 0:
        return clip
    target = float(operation["target_peak"])
    desired = max(1.0, target * PCM_MAX)
    factor = desired / peak
    return replace_channels(
        clip,
        (
            [_clip_sample(value * factor) for value in values]
            for values in clip.channels
        ),
    )


def mix_audio(
    primary: AudioClip,
    secondary: AudioClip,
    operation: dict[str, Any],
) -> AudioClip:
    if primary.sample_rate != secondary.sample_rate:
        raise MediaOperationFailure(
            "AUDIO_MIX_RATE_MISMATCH",
            "Mixed audio sources must share a sample rate.",
        )
    if primary.channel_count != secondary.channel_count:
        raise MediaOperationFailure(
            "AUDIO_MIX_CHANNEL_MISMATCH",
            "Mixed audio sources must share a channel count.",
        )
    primary_weight = float(operation["primary_weight"])
    secondary_weight = float(operation["secondary_weight"])
    secondary_factor = _gain_factor(float(operation["secondary_gain"])) * secondary_weight
    alignment = operation.get("secondary_start_seconds")

    if alignment is None:
        # Legacy behavior: both clips are pinned at their first sample and the
        # output spans the longer clip starting at the current primary head.
        frame_count = max(primary.frame_count, secondary.frame_count)
        mixed: list[list[int]] = []
        for channel_index in range(primary.channel_count):
            left = primary.channels[channel_index]
            right = secondary.channels[channel_index]
            values: list[int] = []
            for frame_index in range(frame_count):
                left_value = left[frame_index] if frame_index < len(left) else 0
                right_value = right[frame_index] if frame_index < len(right) else 0
                combined = left_value * primary_weight + right_value * secondary_factor
                values.append(_clip_sample(combined))
            mixed.append(values)
        return replace_channels(primary, mixed)

    # Alignment-aware behavior. Both recordings are placed on one shared
    # timeline: the original primary recording's timeline. The current primary
    # clip remembers how far its head was trimmed, and the secondary is placed
    # at its recorded start position. The output covers the full union so that
    # secondary-only lead-in/tail and the true overlap all remain audible.
    secondary_origin = int(round(float(alignment) * primary.sample_rate))
    primary_origin = primary.source_start_frame
    primary_end = primary_origin + primary.frame_count
    secondary_end = secondary_origin + secondary.frame_count
    span_start = min(primary_origin, secondary_origin)
    span_end = max(primary_end, secondary_end)
    span = span_end - span_start
    if span <= 0:
        raise MediaOperationFailure(
            "AUDIO_MIX_EMPTY",
            "The aligned audio sources produce no sample frames.",
        )

    mixed = []
    for channel_index in range(primary.channel_count):
        left = primary.channels[channel_index]
        right = secondary.channels[channel_index]
        values = []
        for frame in range(span_start, span_end):
            primary_offset = frame - primary_origin
            left_value = (
                left[primary_offset]
                if 0 <= primary_offset < primary.frame_count
                else 0
            )
            secondary_offset = frame - secondary_origin
            right_value = (
                right[secondary_offset]
                if 0 <= secondary_offset < secondary.frame_count
                else 0
            )
            combined = left_value * primary_weight + right_value * secondary_factor
            values.append(_clip_sample(combined))
        mixed.append(values)
    return replace_channels(
        primary,
        mixed,
        source_start_frame=span_start,
    )


def render_audio(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> bytes:
    clip = decode_wav(primary_bytes)
    secondary = decode_wav(secondary_bytes) if secondary_bytes is not None else None
    mixed = False

    try:
        for operation in operations:
            operation_type = operation["type"]
            if operation_type == "media_trim":
                clip = trim_audio(clip, operation)
            elif operation_type == "audio_gain":
                clip = gain_audio(clip, operation)
            elif operation_type == "audio_normalize":
                clip = normalize_audio(clip, operation)
            elif operation_type == "audio_mix":
                if secondary is None:
                    raise MediaOperationFailure(
                        "AUDIO_MIX_SOURCE_MISSING",
                        "audio_mix requires a secondary media asset.",
                    )
                if mixed:
                    raise MediaOperationFailure(
                        "AUDIO_MIX_DUPLICATE",
                        "An audio plan may mix the secondary source only once.",
                    )
                clip = mix_audio(clip, secondary, operation)
                mixed = True
            else:
                raise MediaOperationFailure(
                    "UNSUPPORTED_AUDIO_OPERATION",
                    f"Unsupported audio operation: {operation_type}",
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "AUDIO_OPERATION_INVALID",
            "The audio operation payload is malformed.",
        ) from exc

    return encode_wav(clip)
