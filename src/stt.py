"""
Speech-to-Text engine using OpenAI Whisper.

Provides local, on-device transcription with language detection and VAD.
"""

import asyncio
import io
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from .config import WhisperConfig

# Suppress FP16 warnings on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


class WhisperSTT:
    """Whisper speech-to-text engine."""

    def __init__(self, config: WhisperConfig):
        """Initialize Whisper STT engine."""
        self.config = config
        self._model = None
        self._device = config.device

        logger.info(f"Initializing Whisper STT (model: {config.model}, device: {config.device})")

    async def initialize(self) -> None:
        """Load Whisper model asynchronously."""
        if self._model is not None:
            return

        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, self._load_model
        )

        logger.info(f"✓ Whisper model '{self.config.model}' loaded on {self._device}")

    def _load_model(self):
        """Load Whisper model (runs in thread pool)."""
        import whisper

        model = whisper.load_model(
            self.config.model,
            device=self._device,
        )
        return model

    async def transcribe(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_data: Audio array (float32, normalized to [-1, 1])
            language: Language code (e.g., 'en'), or None for auto-detect
            task: 'transcribe' or 'translate' (to English)

        Returns:
            Transcribed text
        """
        if self._model is None:
            await self.initialize()

        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio to [-1, 1] if needed
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.abs(audio_data).max()

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_data,
            language or self.config.language,
            task,
        )

        text = result["text"].strip()
        logger.debug(f"Transcribed: '{text}' (language: {result.get('language', 'unknown')})")

        return text

    def _transcribe_sync(
        self,
        audio_data: np.ndarray,
        language: Optional[str],
        task: str,
    ) -> dict:
        """Synchronous transcription (runs in thread pool)."""
        # Prepare options
        options = {
            "task": task,
            "fp16": self._device == "cuda",  # Only use FP16 on CUDA
        }

        if language:
            options["language"] = language

        # Run Whisper
        result = self._model.transcribe(audio_data, **options)

        return result

    async def transcribe_file(self, audio_path: Path, language: Optional[str] = None) -> str:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code or None for auto-detect

        Returns:
            Transcribed text
        """
        # Load audio file
        import soundfile as sf

        audio_data, sample_rate = sf.read(audio_path, dtype="float32")

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio_data = await self._resample_audio(audio_data, sample_rate, 16000)

        return await self.transcribe(audio_data, language)

    async def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy import signal

        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        num_samples = int(len(audio) * ratio)

        # Resample in thread pool
        loop = asyncio.get_event_loop()
        resampled = await loop.run_in_executor(
            None,
            signal.resample,
            audio,
            num_samples,
        )

        return resampled.astype(np.float32)

    def detect_language(self, audio_data: np.ndarray) -> str:
        """
        Detect language from audio.

        Args:
            audio_data: Audio array

        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")

        # Whisper has built-in language detection
        audio_data = audio_data.astype(np.float32)

        # Pad/trim to 30 seconds for language detection
        target_length = 16000 * 30  # 30 seconds at 16kHz
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        else:
            audio_data = audio_data[:target_length]

        # Detect language
        audio_tensor = torch.from_numpy(audio_data).to(self._device)
        _, probs = self._model.detect_language(audio_tensor)

        # Get most likely language
        language = max(probs, key=probs.get)

        logger.debug(f"Detected language: {language} (confidence: {probs[language]:.2f})")

        return language

    async def health_check(self) -> bool:
        """Check if STT engine is healthy."""
        try:
            if self._model is None:
                await self.initialize()

            # Test with silence
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            result = await self.transcribe(test_audio)

            return True
        except Exception as e:
            logger.error(f"STT health check failed: {e}")
            return False

    def close(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            # Clear CUDA cache if using GPU
            if self._device == "cuda":
                torch.cuda.empty_cache()

            self._model = None
            logger.info("Whisper STT closed")
