# LMS Voice AI Agent (LMS-VAI-AGENT)

**Full-stack voice interface** powered by local LLM inference, MCP tools, and on-device speech processing.

## 🎯 Overview

A production-ready voice AI agent that:
- ✅ Accepts **spoken commands** via microphone
- ✅ Transcribes with **Whisper STT** (local, no cloud)
- ✅ Resolves intents with **LMStudio LLM** (local inference)
- ✅ Executes actions via **MCP server tools**
- ✅ Responds with **Coqui TTS** (local, natural voice)
- ✅ Runs **asynchronously** for < 1s latency

### Key Improvements Over Original Outline

1. **Cross-Platform Support** - Optimized for Windows with Linux compatibility
2. **Production-Ready** - Complete error handling, logging, and monitoring
3. **Modular Architecture** - Clean separation of concerns (STT, TTS, LLM, Tools)
4. **Configuration Management** - Centralized config with environment variables
5. **Security First** - Token auth, sandboxed execution, audit logging
6. **Developer Experience** - Type hints, tests, documentation, easy setup

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| **Python** | 3.11+ | Required for async features |
| **LMStudio** | v0.4.1+ | Local LLM inference |
| **Git** | Latest | Version control |
| **Audio Device** | Any | Microphone + speaker/headphones |

**Optional:**
- CUDA-capable GPU (for faster inference)
- Docker Desktop (for containerized deployment)

### Installation

```powershell
# Clone repository
git clone https://github.com/jhuntersatl/LMS-VAI-AGENT.git
cd LMS-VAI-AGENT

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Download Whisper model (first run only)
python -c "import whisper; whisper.load_model('base')"

# Download TTS models (first run only)
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"
```

### First Run

```powershell
# Start LMStudio (must be running first)
# Load your model in LMStudio on http://localhost:1234

# Start the voice agent
python -m src.main

# Or use the CLI
vai-agent start
```

---

## 📁 Project Structure

```
lms-vai-agent/
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point & orchestrator
│   ├── config.py               # Configuration management
│   ├── stt.py                  # Whisper speech-to-text
│   ├── tts.py                  # Coqui text-to-speech
│   ├── lm_client.py            # LMStudio API client
│   ├── mcp_client.py           # MCP server client
│   ├── audio_manager.py        # Audio I/O handling
│   ├── intent_parser.py        # Intent recognition
│   └── utils/
│       ├── logging.py          # Centralized logging
│       └── validators.py       # Input validation
├── config/
│   ├── agent.yaml              # Agent behavior config
│   ├── mcp_tools.yaml          # MCP tool definitions
│   └── prompts/
│       ├── system.txt          # System prompt
│       └── intent.txt          # Intent parsing prompt
├── tests/
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_lm_client.py
│   ├── test_orchestrator.py
│   └── fixtures/
│       └── sample_audio.wav    # Test audio files
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── scripts/
│   ├── install.ps1             # Windows installer
│   ├── install.sh              # Linux installer
│   ├── start_service.ps1       # Windows service script
│   └── benchmark.py            # Performance testing
├── docs/
│   ├── ARCHITECTURE.md         # System design
│   ├── API.md                  # API documentation
│   ├── DEPLOYMENT.md           # Deployment guide
│   └── TROUBLESHOOTING.md      # Common issues
├── .env.example                # Environment template
├── .gitignore
├── pyproject.toml              # Project metadata
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── LICENSE
```

---

## 🏗️ Architecture

### System Flow

```
┌─────────────┐
│ Microphone  │
└──────┬──────┘
       │ Audio Stream
       ▼
┌─────────────────┐
│  Audio Manager  │ ← sounddevice
│   (VAD + Buffer)│
└──────┬──────────┘
       │ Audio Chunks
       ▼
┌─────────────────┐
│   Whisper STT   │ ← openai-whisper
│  (Transcription)│
└──────┬──────────┘
       │ Text
       ▼
┌─────────────────┐
│ Intent Parser   │ ← LMStudio
│  (+ LLM Call)   │
└──────┬──────────┘
       │ Intent + Params
       ▼
┌─────────────────┐
│   MCP Client    │ ← HTTP/JSON
│  (Tool Executor)│
└──────┬──────────┘
       │ Tool Result
       ▼
┌─────────────────┐
│  LMStudio LLM   │ ← Generate response
│  (Synthesis)    │
└──────┬──────────┘
       │ Response Text
       ▼
┌─────────────────┐
│   Coqui TTS     │ ← Text-to-speech
│  (Audio Gen)    │
└──────┬──────────┘
       │ Audio
       ▼
┌─────────────────┐
│    Speaker      │
└─────────────────┘
```

### Key Components

1. **Audio Manager** (`audio_manager.py`)
   - Voice Activity Detection (VAD)
   - 2-second buffering
   - Cross-platform audio I/O

2. **STT Engine** (`stt.py`)
   - Local Whisper model (base/small/medium)
   - Async transcription
   - Language auto-detection

3. **Intent Parser** (`intent_parser.py`)
   - LLM-powered intent extraction
   - Parameter validation
   - Tool routing logic

4. **LM Client** (`lm_client.py`)
   - LMStudio HTTP API wrapper
   - Streaming response support
   - Token management

5. **MCP Client** (`mcp_client.py`)
   - Tool discovery
   - Secure execution
   - Result formatting

6. **TTS Engine** (`tts.py`)
   - Coqui TTS models
   - Audio streaming
   - Speed/pitch control

7. **Orchestrator** (`main.py`)
   - AsyncIO event loop
   - Pipeline coordination
   - Error handling & retry logic

---

## ⚙️ Configuration

### Environment Variables

See [.env.example](.env.example) for all configuration options.

**Key Settings:**

```env
# LMStudio
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_MODEL=llama-3-70b

# MCP Server
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_api_key_here

# Whisper
WHISPER_MODEL=base  # tiny|base|small|medium|large
WHISPER_DEVICE=cpu  # cpu|cuda|mps

# TTS
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
TTS_DEVICE=cpu  # cpu|cuda
```

### YAML Configuration

**config/agent.yaml:**
```yaml
agent:
  name: "VAI Agent"
  wake_word: ""  # Optional wake word
  max_history: 10
  response_timeout: 30

audio:
  sample_rate: 16000
  chunk_size: 1024
  vad_threshold: 0.5
  silence_duration: 2.0
```

---

## 🔧 Development

### Running Tests

```powershell
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_stt.py -v

# Async tests
pytest -k test_orchestrator --asyncio-mode=auto
```

### Code Quality

```powershell
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Debugging

```powershell
# Enable debug logging
$env:LOG_LEVEL="DEBUG"
$env:DEBUG="true"
python -m src.main

# Profile performance
python scripts/benchmark.py
```

---

## 🐳 Docker Deployment

### Quick Start

```powershell
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f vai-agent

# Stop
docker-compose down
```

### Manual Docker

```powershell
# Build
docker build -t lms-vai-agent:latest -f docker/Dockerfile .

# Run (requires host audio access)
docker run -it --rm \
  --device /dev/snd \
  -e LMSTUDIO_BASE_URL=http://host.docker.internal:1234 \
  --env-file .env \
  lms-vai-agent:latest
```

**Windows Note:** Docker audio access requires WSL2 backend with PulseAudio.

---

## 🛡️ Security

### Best Practices

1. **MCP Tool Sandboxing**
   - Use Docker containers for tool execution
   - Restrict file system access
   - Whitelist allowed commands

2. **Authentication**
   - Set `MCP_API_KEY` for MCP server
   - Use TLS for remote LMStudio
   - Rotate keys regularly

3. **Audit Logging**
   - All tool executions logged
   - User commands tracked
   - Error events recorded

4. **Input Validation**
   - Sanitize file paths
   - Validate URLs
   - Escape shell commands

See [docs/SECURITY.md](docs/SECURITY.md) for detailed guidelines.

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **End-to-End Latency** | < 1 second | Voice → response |
| **STT Latency** | < 200ms | Base model |
| **LLM Inference** | < 500ms | 70B model on GPU |
| **TTS Generation** | < 300ms | Streaming mode |
| **Memory Usage** | < 8 GB | Without model cache |

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 12+ cores |
| **RAM** | 16 GB | 32 GB |
| **GPU** | None | NVIDIA RTX 4090 (24GB VRAM) |
| **Storage** | 20 GB | 100 GB SSD |

---

## 🚧 Roadmap

### Phase 1: Core Functionality ✅
- [x] STT with Whisper
- [x] TTS with Coqui
- [x] LMStudio integration
- [x] MCP client
- [x] Basic orchestration

### Phase 2: Production Ready (Current)
- [ ] Error handling & retry logic
- [ ] Comprehensive testing
- [ ] Docker deployment
- [ ] Documentation

### Phase 3: Advanced Features
- [ ] Wake word detection
- [ ] Multi-language support
- [ ] Voice cloning
- [ ] Emotion detection
- [ ] Context awareness

### Phase 4: Scale & Polish
- [ ] Multi-speaker support
- [ ] WebSocket API
- [ ] Web dashboard
- [ ] Plugin system
- [ ] Mobile app

---

## 🐛 Troubleshooting

### Common Issues

**Audio not working:**
```powershell
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
python scripts/test_audio.py
```

**LMStudio connection failed:**
```powershell
# Check if LMStudio is running
curl http://localhost:1234/v1/models

# Verify model loaded
# Open LMStudio GUI → Server tab → Ensure model is active
```

**Whisper model download fails:**
```powershell
# Manual download
python -c "import whisper; whisper.load_model('base', download_root='./models')"
```

**Out of memory:**
- Use smaller Whisper model (tiny/base)
- Reduce TTS buffer size
- Enable model offloading in LMStudio

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Additional Resources

- [LMStudio Documentation](https://lmstudio.ai/docs)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [MCP Protocol Spec](https://modelcontextprotocol.org)

---

## 🙏 Acknowledgments

- OpenAI for Whisper
- Coqui for TTS
- LMStudio team
- MCP community

---

**Built with ❤️ for local-first AI**
