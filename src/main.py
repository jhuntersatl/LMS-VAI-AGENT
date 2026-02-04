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
        self.conversation_history = []

        # Components (lazy-loaded)
        self._lm_client = None
        self._stt_engine = None
        self._tts_engine = None
        self._mcp_client = None
        self._audio_manager = None
        self._intent_parser = None

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
        from .audio_manager import AudioManager
        from .intent_parser import IntentParser
        from .lm_client import LMStudioClient
        from .mcp_client import MCPClient
        from .stt import WhisperSTT
        from .tts import CoquiTTS

        # Initialize LMStudio
        self._lm_client = LMStudioClient(self.config.lmstudio)
        if not await self._lm_client.health_check():
            logger.error("LMStudio is not available. Please start LMStudio first.")
            raise RuntimeError("LMStudio not available")
        logger.info("✓ LMStudio connected")

        # Initialize MCP client
        self._mcp_client = MCPClient(self.config.mcp)
        await self._mcp_client.initialize()

        # Initialize STT
        self._stt_engine = WhisperSTT(self.config.whisper)
        await self._stt_engine.initialize()

        # Initialize TTS
        self._tts_engine = CoquiTTS(self.config.tts)
        await self._tts_engine.initialize()

        # Initialize Audio Manager
        self._audio_manager = AudioManager(self.config.audio)

        # Initialize Intent Parser
        self._intent_parser = IntentParser(self._lm_client, self._mcp_client)

        logger.info("✅ Voice AI Agent initialized successfully")

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
        try:
            # 1. Listen for audio with VAD
            console.print("[yellow]🎤 Listening...[/yellow]")
            audio_data = await self._audio_manager.record_until_silence(
                max_duration=self.config.agent.response_timeout
            )

            # Check if audio is meaningful
            if len(audio_data) < 1000:  # Too short
                await asyncio.sleep(0.1)
                return

            # 2. Transcribe with Whisper
            console.print("[cyan]🔄 Transcribing...[/cyan]")
            text = await self._stt_engine.transcribe(audio_data)

            if not text or len(text.strip()) < 2:
                logger.debug("No speech detected")
                return

            console.print(f"[bold]You:[/bold] {text}")
            self.conversation_history.append({"role": "user", "content": text})

            # 3. Parse intent and execute tools
            console.print("[cyan]🧠 Understanding...[/cyan]")
            execution_result = await self._intent_parser.parse_and_execute(text)

            # 4. Generate response
            console.print("[cyan]💭 Thinking...[/cyan]")
            response_text = await self._generate_response(text, execution_result)

            console.print(f"[bold green]Agent:[/bold green] {response_text}")
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Trim history
            if len(self.conversation_history) > self.config.agent.max_history * 2:
                self.conversation_history = self.conversation_history[-self.config.agent.max_history * 2:]

            # 5. Synthesize and speak response
            console.print("[cyan]🔊 Speaking...[/cyan]")
            await self._tts_engine.speak(response_text)

            console.print("")  # Empty line for spacing

        except Exception as e:
            logger.error(f"Error in voice loop: {e}", exc_info=True)
            await asyncio.sleep(1)

    async def _generate_response(
        self,
        user_input: str,
        execution_result: dict,
    ) -> str:
        """
        Generate natural language response.

        Args:
            user_input: User's original input
            execution_result: Result from intent execution

        Returns:
            Response text to speak
        """
        # Build context from execution result
        if execution_result.get("is_conversation", True):
            # Simple conversation
            system_prompt = "You are a helpful voice AI assistant. Respond naturally and concisely."
            prompt = user_input

        else:
            # Tool was executed
            tool_name = execution_result.get("tool", "unknown")
            success = execution_result.get("success", False)
            result_data = execution_result.get("result", "")
            error = execution_result.get("error")

            if success:
                system_prompt = f"The user asked: '{user_input}'. The tool '{tool_name}' was executed successfully. Summarize the result naturally."
                prompt = f"Tool result: {result_data}"
            else:
                system_prompt = "The requested action failed. Explain the error to the user politely."
                prompt = f"Error: {error}"

        # Generate response
        response = await self._lm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=150,
        )

        return response.strip()

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Shutting down Voice AI Agent...")
        self.running = False

        if self._audio_manager:
            self._audio_manager.close()

        if self._stt_engine:
            self._stt_engine.close()

        if self._tts_engine:
            self._tts_engine.close()

        if self._mcp_client:
            await self._mcp_client.close()

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
