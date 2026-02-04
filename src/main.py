"""
Main orchestrator and CLI entry point for LMS Voice AI Agent.
"""

import asyncio
import signal
import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from .config import get_config

console = Console()


class VoiceAgent:
    """Main voice agent orchestrator."""

    def __init__(self):
        """Initialize voice agent."""
        self.config = get_config()
        self.running = False

        # Components (lazy-loaded)
        self._lm_client = None
        self._stt_engine = None
        self._tts_engine = None
        self._mcp_client = None
        self._audio_manager = None

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Voice AI Agent...")

        # Setup logging
        log_path = self.config.logs_dir / "vai-agent.log"
        logger.add(
            log_path,
            rotation=self.config.logging.log_max_size,
            retention=self.config.logging.log_backup_count,
            level=self.config.logging.log_level,
        )

        # Import and initialize components
        from .lm_client import LMStudioClient

        self._lm_client = LMStudioClient(self.config.lmstudio)

        # Check LMStudio health
        if not await self._lm_client.health_check():
            logger.error("LMStudio is not available. Please start LMStudio first.")
            raise RuntimeError("LMStudio not available")

        logger.info("✓ LMStudio connected")

        # TODO: Initialize other components
        # self._stt_engine = WhisperSTT(self.config.whisper)
        # self._tts_engine = CoquiTTS(self.config.tts)
        # self._mcp_client = MCPClient(self.config.mcp)
        # self._audio_manager = AudioManager(self.config.audio)

        logger.info("Voice AI Agent initialized successfully")

    async def start(self) -> None:
        """Start the voice agent main loop."""
        self.running = True
        logger.info("Voice AI Agent starting...")

        console.print(
            Panel.fit(
                "[bold green]🎤 Voice AI Agent Active[/bold green]\\n"
                "Speak to interact. Press Ctrl+C to stop.",
                border_style="green",
            )
        )

        try:
            while self.running:
                await self._process_voice_loop()

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.shutdown()

    async def _process_voice_loop(self) -> None:
        """Main voice processing loop."""
        # TODO: Implement full pipeline
        # 1. Listen for audio (Audio Manager + VAD)
        # 2. Transcribe (Whisper STT)
        # 3. Parse intent (LMStudio)
        # 4. Execute tool (MCP Client)
        # 5. Generate response (LMStudio)
        # 6. Speak response (Coqui TTS)

        # Placeholder: Just sleep for now
        await asyncio.sleep(1)

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Shutting down Voice AI Agent...")
        self.running = False

        # TODO: Cleanup components
        # if self._audio_manager:
        #     await self._audio_manager.close()

        logger.info("Voice AI Agent stopped")


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LMS Voice AI Agent - Local voice interface with LLM."""
    pass


@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def start(debug: bool):
    """Start the voice agent service."""
    if debug:
        import os

        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"

    async def run():
        agent = VoiceAgent()
        await agent.initialize()
        await agent.start()

    try:
        asyncio.run(run())
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        sys.exit(1)


@cli.command()
def test():
    """Test system components."""
    console.print("[bold cyan]Testing LMS Voice AI Agent components...[/bold cyan]")

    async def run_tests():
        config = get_config()

        # Test LMStudio
        console.print("\\n[yellow]Testing LMStudio connection...[/yellow]")
        from .lm_client import LMStudioClient

        lm = LMStudioClient(config.lmstudio)
        if await lm.health_check():
            console.print("[green]✓ LMStudio is available[/green]")

            # Test generation
            response = await lm.generate("Say 'hello'", max_tokens=20)
            console.print(f"[green]✓ Generated response: {response}[/green]")
        else:
            console.print("[red]✗ LMStudio is not available[/red]")

        # TODO: Test other components
        console.print("\\n[green]Component tests complete![/green]")

    asyncio.run(run_tests())


@cli.command()
def config_info():
    """Display current configuration."""
    config = get_config()

    console.print(
        Panel.fit(
            f"[bold]LMStudio:[/bold] {config.lmstudio.base_url}\\n"
            f"[bold]Model:[/bold] {config.lmstudio.model}\\n"
            f"[bold]Whisper:[/bold] {config.whisper.model} on {config.whisper.device}\\n"
            f"[bold]TTS:[/bold] {config.tts.device}\\n"
            f"[bold]Log Level:[/bold] {config.logging.log_level}",
            title="Configuration",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    cli()
