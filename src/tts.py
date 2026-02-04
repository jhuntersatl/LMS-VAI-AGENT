"""
Text-to-Speech engine using Coqui TTS.

Provides local, on-device speech synthesis with multiple voice options.
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from .config import TTSConfig


class CoquiTTS:
    """Coqui TTS speech synthesis engine."""

    def __init__(self, config: TTSConfig):
        """Initialize Coqui TTS engine."""
        self.config = config
        self._tts = None
        self._device = config.device

        logger.info(f"Initializing Coqui TTS (model: {config.model}, device: {config.device})")

    async def initialize(self) -> None:
        """Load TTS model asynchronously."""
        if self._tts is not None:
            return

        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._tts = await loop.run_in_executor(
            None, self._load_model
        )

        logger.info(f"✓ Coqui TTS model loaded on {self._device}")

    def _load_model(self):
        """Load TTS model (runs in thread pool)."""
        from TTS.api import TTS

        # Initialize TTS with model name
        tts = TTS(
            model_name=self.config.model,
            gpu=(self._device == "cuda"),
        )

        return tts

    async def synthesize(
        self,
        text: str,
        speed: Optional[float] = None,
        save_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speed: Speech speed multiplier (default: from config)
            save_path: Optional path to save audio file

        Returns:
            Audio array (float32, sample rate from model)
        """
        if self._tts is None:
            await self.initialize()

        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return np.zeros(1000, dtype=np.float32)

        speed = speed or self.config.speed

        logger.debug(f"Synthesizing: '{text[:50]}...' (speed: {speed})")

        # Run synthesis in thread pool
        loop = asyncio.get_event_loop()

        if save_path:
            # Synthesize directly to file
            await loop.run_in_executor(
                None,
                self._synthesize_to_file,
                text,
                save_path,
            )

            # Load the audio back
            audio_data, sample_rate = sf.read(save_path, dtype="float32")
        else:
            # Synthesize to numpy array
            audio_data = await loop.run_in_executor(
                None,
                self._synthesize_to_array,
                text,
            )

        # Apply speed adjustment if needed
        if speed != 1.0:
            audio_data = await self._adjust_speed(audio_data, speed)

        logger.debug(f"Synthesized {len(audio_data)} samples")

        return audio_data

    def _synthesize_to_array(self, text: str) -> np.ndarray:
        """Synthesize to numpy array (runs in thread pool)."""
        # Create temporary file for Coqui TTS
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Synthesize to temp file
            self._tts.tts_to_file(
                text=text,
                file_path=tmp_path,
            )

            # Load audio from temp file
            audio_data, sample_rate = sf.read(tmp_path, dtype="float32")

            return audio_data

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def _synthesize_to_file(self, text: str, output_path: Path) -> None:
        """Synthesize directly to file (runs in thread pool)."""
        self._tts.tts_to_file(
            text=text,
            file_path=str(output_path),
        )

    async def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Adjust audio playback speed.

        Args:
            audio: Audio array
            speed: Speed multiplier (0.5 = slower, 2.0 = faster)

        Returns:
            Speed-adjusted audio
        """
        from scipy import signal

        # Calculate new length
        new_length = int(len(audio) / speed)

        # Resample in thread pool
        loop = asyncio.get_event_loop()
        resampled = await loop.run_in_executor(
            None,
            signal.resample,
            audio,
            new_length,
        )

        return resampled.astype(np.float32)

    async def speak(
        self,
        text: str,
        speed: Optional[float] = None,
        blocking: bool = True,
    ) -> None:
        """
        Synthesize and play audio through speaker.

        Args:
            text: Text to speak
            speed: Speech speed multiplier
            blocking: Wait for playback to complete
        """
        audio_data = await self.synthesize(text, speed)

        # Get sample rate from TTS model
        sample_rate = self._get_sample_rate()

        # Play audio
        if blocking:
            sd.play(audio_data, sample_rate)
            sd.wait()
        else:
            sd.play(audio_data, sample_rate)

        logger.info(f"Played audio: '{text[:50]}...'")

    def _get_sample_rate(self) -> int:
        """Get sample rate from TTS model."""
        # Most Coqui TTS models use 22050 Hz
        # This can be queried from the model if needed
        return getattr(self._tts, "sample_rate", 22050)

    async def synthesize_batch(
        self,
        texts: list[str],
        speed: Optional[float] = None,
    ) -> list[np.ndarray]:
        """
        Synthesize multiple texts in batch.

        Args:
            texts: List of texts to synthesize
            speed: Speech speed multiplier

        Returns:
            List of audio arrays
        """
        tasks = [self.synthesize(text, speed) for text in texts]
        return await asyncio.gather(*tasks)

    async def stream_synthesize(
        self,
        text_iterator,
        speed: Optional[float] = None,
    ):
        """
        Synthesize text as it arrives (streaming mode).

        Args:
            text_iterator: Async iterator yielding text chunks
            speed: Speech speed multiplier

        Yields:
            Audio chunks as they're synthesized
        """
        async for text_chunk in text_iterator:
            if text_chunk.strip():
                audio = await self.synthesize(text_chunk, speed)
                yield audio

    async def health_check(self) -> bool:
        """Check if TTS engine is healthy."""
        try:
            if self._tts is None:
                await self.initialize()

            # Test with short phrase
            test_audio = await self.synthesize("Test.")

            return len(test_audio) > 0

        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return False

    def close(self) -> None:
        """Clean up resources."""
        if self._tts is not None:
            # Stop any playing audio
            sd.stop()

            # Clear model
            self._tts = None

            logger.info("Coqui TTS closed")

    def list_available_models(self) -> list[str]:
        """List available TTS models."""
        from TTS.api import TTS

        return TTS.list_models()

    def get_model_info(self) -> dict:
        """Get information about loaded model."""
        if self._tts is None:
            return {}

        return {
            "model_name": self.config.model,
            "device": self._device,
            "sample_rate": self._get_sample_rate(),
            "speed": self.config.speed,
        }
