"""
Configuration management for LMS Voice AI Agent.

Loads settings from environment variables and config files.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LMStudioConfig(BaseSettings):
    """LMStudio API configuration."""

    base_url: str = Field(default="http://localhost:1234", env="LMSTUDIO_BASE_URL")
    model: str = Field(default="llama-3-70b", env="LMSTUDIO_MODEL")
    api_key: Optional[str] = Field(default=None, env="LMSTUDIO_API_KEY")
    timeout: int = Field(default=30, env="LMSTUDIO_TIMEOUT")


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    server_url: str = Field(default="http://localhost:8000", env="MCP_SERVER_URL")
    api_key: Optional[str] = Field(default=None, env="MCP_API_KEY")
    timeout: int = Field(default=10, env="MCP_TIMEOUT")


class WhisperConfig(BaseSettings):
    """Whisper STT configuration."""

    model: str = Field(default="base", env="WHISPER_MODEL")
    device: str = Field(default="cpu", env="WHISPER_DEVICE")
    language: Optional[str] = Field(default="en", env="WHISPER_LANGUAGE")


class TTSConfig(BaseSettings):
    """Coqui TTS configuration."""

    model: str = Field(
        default="tts_models/en/ljspeech/tacotron2-DDC",
        env="TTS_MODEL"
    )
    vocoder: Optional[str] = Field(
        default="vocoder_models/en/ljspeech/hifigan_v2",
        env="TTS_VOCODER"
    )
    device: str = Field(default="cpu", env="TTS_DEVICE")
    speed: float = Field(default=1.0, env="TTS_SPEED")


class AudioConfig(BaseSettings):
    """Audio processing configuration."""

    sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    chunk_size: int = Field(default=1024, env="AUDIO_CHUNK_SIZE")
    vad_threshold: float = Field(default=0.5, env="VAD_THRESHOLD")
    silence_duration: float = Field(default=2.0, env="SILENCE_DURATION")


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    wake_word: Optional[str] = Field(default=None, env="WAKE_WORD")
    max_history: int = Field(default=10, env="MAX_HISTORY")
    response_timeout: int = Field(default=30, env="RESPONSE_TIMEOUT")
    debug: bool = Field(default=False, env="DEBUG")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/vai-agent.log", env="LOG_FILE")
    log_max_size: str = Field(default="10MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class PerformanceConfig(BaseSettings):
    """Performance tuning configuration."""

    worker_threads: int = Field(default=4, env="WORKER_THREADS")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    cache_dir: str = Field(default="./models", env="CACHE_DIR")


class Config(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )

    # Sub-configs
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @property
    def config_dir(self) -> Path:
        """Get config directory."""
        return self.project_root / "config"

    @property
    def models_dir(self) -> Path:
        """Get models cache directory."""
        path = Path(self.performance.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        path = self.project_root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global _config
    _config = Config()
    return _config
