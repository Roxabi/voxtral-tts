"""Voxtral-4B-TTS local inference with int4 HQQ quantization."""

from voxtral_tts.model import VoxtralTTS, VoxtralConfig
from voxtral_tts.generate import TekkenTokenizer
from voxtral_tts.generate_fast import generate_speech_fast, enable_static_cache, reset_static_cache
from voxtral_tts.load_model import load_original_model
from voxtral_tts.audio_postprocess import postprocess_audio, trim_warmup_frames

__all__ = [
    "VoxtralTTS",
    "VoxtralConfig",
    "TekkenTokenizer",
    "generate_speech_fast",
    "enable_static_cache",
    "reset_static_cache",
    "load_original_model",
    "postprocess_audio",
    "trim_warmup_frames",
    "load_model_int4",
]


def load_model_int4(model_dir: str, device: str = "cuda", group_size: int = 64) -> VoxtralTTS:
    """Load Voxtral-4B-TTS with int4 quantized backbone.

    Lazy import to avoid hard dependency on torchao/hqq when using BF16 mode.
    Install with: pip install voxtral-tts[int4]
    """
    from voxtral_tts.torchao_inference import load_model_int4 as _load

    return _load(model_dir, device=device, group_size=group_size)
