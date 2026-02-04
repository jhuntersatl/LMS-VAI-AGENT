"""
Audio input/output manager with Voice Activity Detection (VAD).

Handles microphone input, speaker output, and voice detection.
"""

import asyncio
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from .config import AudioConfig


class AudioManager:
    """Audio I/O manager with VAD and buffering."""

    def __init__(self, config: AudioConfig):
        """Initialize audio manager."""
        self.config = config
        self.sample_rate = config.sample_rate
        self.chunk_size = config.chunk_size
        self.vad_threshold = config.vad_threshold
        self.silence_duration = config.silence_duration

        # Audio buffers
        self._audio_queue: queue.Queue = queue.Queue()
        self._recording = False
        self._stream: Optional[sd.InputStream] = None

        # VAD state
        self._is_speaking = False
        self._silence_start = None
        self._speech_buffer = []

        logger.info(f"Audio Manager initialized (sr: {self.sample_rate}, chunk: {self.chunk_size})")

    def start_recording(self, callback: Optional[Callable] = None) -> None:
        """
        Start recording from microphone.

        Args:
            callback: Optional callback for audio chunks (audio_data, sample_rate)
        """
        if self._recording:
            logger.warning("Already recording")
            return

        self._recording = True
        self._speech_buffer = []

        def audio_callback(indata, frames, time_info, status):
            """Audio input callback."""
            if status:
                logger.warning(f"Audio callback status: {status}")

            # Copy audio data (sounddevice reuses buffer)
            audio_chunk = indata.copy().flatten()

            # Put in queue for processing
            self._audio_queue.put(audio_chunk)

            # Call user callback if provided
            if callback:
                callback(audio_chunk, self.sample_rate)

        # Start audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        )

        self._stream.start()
        logger.info("✓ Recording started")

    def stop_recording(self) -> None:
        """Stop recording."""
        if not self._recording:
            return

        self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        logger.info("Recording stopped")

    async def record_until_silence(
        self,
        max_duration: float = 30.0,
    ) -> np.ndarray:
        """
        Record audio until silence is detected.

        Args:
            max_duration: Maximum recording duration in seconds

        Returns:
            Recorded audio as numpy array
        """
        logger.info("Recording until silence...")

        audio_buffer = []
        silence_counter = 0
        max_silence_chunks = int(
            (self.silence_duration * self.sample_rate) / self.chunk_size
        )

        # Start recording
        self.start_recording()

        # Record until silence or max duration
        start_time = asyncio.get_event_loop().time()

        try:
            while True:
                # Check max duration
                if asyncio.get_event_loop().time() - start_time > max_duration:
                    logger.info("Max duration reached")
                    break

                # Get audio chunk (non-blocking)
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                audio_buffer.append(chunk)

                # Voice Activity Detection
                if self._is_voice_active(chunk):
                    silence_counter = 0
                else:
                    silence_counter += 1

                    # Check if silence duration exceeded
                    if silence_counter >= max_silence_chunks and len(audio_buffer) > 10:
                        logger.info(f"Silence detected after {len(audio_buffer)} chunks")
                        break

                await asyncio.sleep(0.001)  # Yield control

        finally:
            self.stop_recording()

        # Concatenate audio chunks
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer)
            logger.info(f"Recorded {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.2f}s)")
            return audio_data
        else:
            logger.warning("No audio recorded")
            return np.zeros(1000, dtype=np.float32)

    def _is_voice_active(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if voice is active in audio chunk using energy-based VAD.

        Args:
            audio_chunk: Audio data

        Returns:
            True if voice detected
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk**2))

        # Simple threshold-based VAD
        is_active = energy > self.vad_threshold

        return is_active

    async def play_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        """
        Play audio through speaker.

        Args:
            audio_data: Audio array to play
            sample_rate: Sample rate (default: from config)
            blocking: Wait for playback to complete
        """
        sample_rate = sample_rate or self.sample_rate

        if blocking:
            sd.play(audio_data, sample_rate)
            sd.wait()
        else:
            sd.play(audio_data, sample_rate)

        logger.debug(f"Played {len(audio_data)} samples at {sample_rate} Hz")

    def stop_playback(self) -> None:
        """Stop current audio playback."""
        sd.stop()
        logger.debug("Playback stopped")

    def list_devices(self) -> list[dict]:
        """
        List available audio devices.

        Returns:
            List of device info dicts
        """
        devices = sd.query_devices()
        logger.info(f"Found {len(devices)} audio devices")

        return [
            {
                "index": i,
                "name": dev["name"],
                "channels_in": dev["max_input_channels"],
                "channels_out": dev["max_output_channels"],
                "sample_rate": dev["default_samplerate"],
            }
            for i, dev in enumerate(devices)
        ]

    def set_device(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ) -> None:
        """
        Set audio input/output devices.

        Args:
            input_device: Input device index
            output_device: Output device index
        """
        if input_device is not None:
            sd.default.device[0] = input_device
            logger.info(f"Set input device to index {input_device}")

        if output_device is not None:
            sd.default.device[1] = output_device
            logger.info(f"Set output device to index {output_device}")

    def get_current_device(self) -> dict:
        """Get current audio device configuration."""
        devices = sd.query_devices()
        input_idx = sd.default.device[0]
        output_idx = sd.default.device[1]

        return {
            "input": devices[input_idx] if input_idx is not None else None,
            "output": devices[output_idx] if output_idx is not None else None,
        }

    def calibrate_vad_threshold(self, duration: float = 3.0) -> float:
        """
        Calibrate VAD threshold based on ambient noise.

        Args:
            duration: Calibration duration in seconds

        Returns:
            Recommended threshold value
        """
        logger.info(f"Calibrating VAD threshold for {duration}s...")
        logger.info("Please remain silent during calibration.")

        energy_samples = []

        def callback(audio_chunk, sr):
            energy = np.sqrt(np.mean(audio_chunk**2))
            energy_samples.append(energy)

        # Record ambient noise
        self.start_recording(callback)

        import time
        time.sleep(duration)

        self.stop_recording()

        # Calculate threshold as 2x mean ambient energy
        if energy_samples:
            mean_energy = np.mean(energy_samples)
            recommended_threshold = mean_energy * 2.0

            logger.info(f"Ambient noise: {mean_energy:.4f}")
            logger.info(f"Recommended threshold: {recommended_threshold:.4f}")

            return recommended_threshold
        else:
            logger.warning("Calibration failed, using default threshold")
            return self.vad_threshold

    async def record_with_timeout(
        self,
        duration: float,
    ) -> np.ndarray:
        """
        Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio
        """
        logger.info(f"Recording for {duration}s...")

        audio_buffer = []
        target_chunks = int((duration * self.sample_rate) / self.chunk_size)

        self.start_recording()

        try:
            for _ in range(target_chunks):
                try:
                    chunk = self._audio_queue.get(timeout=1.0)
                    audio_buffer.append(chunk)
                except queue.Empty:
                    break

                await asyncio.sleep(0.001)

        finally:
            self.stop_recording()

        if audio_buffer:
            return np.concatenate(audio_buffer)
        else:
            return np.zeros(1000, dtype=np.float32)

    def clear_buffer(self) -> None:
        """Clear audio queue buffer."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.debug("Audio buffer cleared")

    def close(self) -> None:
        """Clean up resources."""
        self.stop_recording()
        self.stop_playback()
        self.clear_buffer()

        logger.info("Audio Manager closed")
