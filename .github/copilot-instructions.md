<!-- LMS Voice AI Agent Project Setup Instructions -->

- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
	<!-- Python voice AI agent with LMStudio, MCP tools, Whisper STT, Coqui TTS. Cross-platform (Windows/Linux). -->

- [x] Scaffold the Project
	<!-- Created Python project structure with src/, config/, tests/, docker/, scripts/, docs/ folders -->

- [x] Customize the Project
	<!-- Implemented core modules: config.py, lm_client.py, main.py with async orchestration -->

- [ ] Install Required Extensions
	<!-- Python extension for VS Code -->

- [ ] Compile the Project
	<!-- Install dependencies via pip: pip install -r requirements.txt -->

- [ ] Create and Run Task
	<!-- Create run/test tasks for the voice agent -->

- [ ] Launch the Project
	<!-- Run the voice agent service: python -m src.main start -->

- [x] Ensure Documentation is Complete
	<!-- README.md with comprehensive setup instructions and improved architecture -->

## Project Specifications
- **Type**: Python voice AI agent
- **Platform**: Windows primary, Linux support
- **LLM**: LMStudio local inference
- **STT**: Whisper (OpenAI)
- **TTS**: Coqui TTS
- **Tools**: MCP server integration
- **Async**: Python AsyncIO orchestration

## Repository
https://github.com/jhuntersatl/LMS-VAI-AGENT.git

## Next Steps
1. Install Python dependencies
2. Configure .env file
3. Implement remaining modules (stt.py, tts.py, mcp_client.py, audio_manager.py)
4. Add comprehensive tests
5. Create Docker deployment configuration

