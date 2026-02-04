# Deployment Guide - GMKTEC Evo-X2

Production deployment instructions for the GMKTEC Evo-X2 (128GB VRAM) system.

## System Specifications

- **Platform**: AMD64/x64 Windows 11
- **CPU**: AMD Ryzen (12+ cores)
- **GPU**: NVIDIA (128GB VRAM)
- **RAM**: 128 GB
- **Storage**: 1TB+ NVMe SSD

## Pre-Deployment Checklist

### 1. GPU Drivers
```powershell
# Verify NVIDIA drivers installed
nvidia-smi

# Should show GPU with ~128GB memory
```

### 2. Python Environment
```powershell
# Install Python 3.11 or 3.12 (NOT 3.14 if unstable)
winget install Python.Python.3.11

# Verify
python --version  # Should show 3.11.x
```

### 3. LMStudio Setup
1. Download LMStudio v0.4.1+
2. Install to `C:\Program Files\LMStudio`
3. Load your preferred model (e.g., Llama-3-70B)
4. Start server on port 1234
5. Test: `curl http://localhost:1234/v1/models`

## Installation Steps

### 1. Clone Repository
```powershell
cd C:\Projects
git clone https://github.com/jhuntersatl/LMS-VAI-AGENT.git
cd LMS-VAI-AGENT
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# For GPU acceleration (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configure Environment
```powershell
# Copy template
Copy-Item .env.example .env

# Edit with your settings
notepad .env
```

**Key settings for Evo-X2:**
```env
# Use GPU for faster inference
WHISPER_DEVICE=cuda
TTS_DEVICE=cuda

# Larger model for better quality
WHISPER_MODEL=medium  # or large

# Adjust for your GPU memory
USE_GPU=true
CACHE_DIR=C:/Models/vai-agent
```

### 5. Download Models
```powershell
# Download Whisper model
python -c "import whisper; whisper.load_model('medium', download_root='C:/Models/vai-agent')"

# Download TTS models (first run will auto-download)
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"
```

### 6. Test Audio Devices
```powershell
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set correct device in .env if needed
```

## Running the Agent

### Test Components
```powershell
# Test LMStudio connection
python -m src.main test

# Check configuration
python -m src.main config-info
```

### Start Voice Agent
```powershell
# Normal mode
python -m src.main start

# Debug mode
python -m src.main start --debug
```

## Performance Optimization

### GPU Memory
With 128GB VRAM, you can:
- Run large Whisper models (large-v3)
- Use multiple TTS voices simultaneously
- Load larger LLM models in LMStudio
- Enable batch processing

### Recommended Settings
```env
# .env for Evo-X2
WHISPER_MODEL=large-v3
TTS_DEVICE=cuda
WORKER_THREADS=16  # Adjust to CPU cores
USE_GPU=true
```

### LMStudio Configuration
- Model: Llama-3-70B or larger
- Context length: 8192+
- GPU layers: Max (offload all to VRAM)
- Batch size: 512

## Running as Service

### Option 1: Task Scheduler
```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "C:\Projects\LMS-VAI-AGENT\venv\Scripts\python.exe" -Argument "-m src.main start" -WorkingDirectory "C:\Projects\LMS-VAI-AGENT"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName "VoiceAIAgent" -Action $action -Trigger $trigger -Principal $principal
```

### Option 2: NSSM (Non-Sucking Service Manager)
```powershell
# Install NSSM
winget install NSSM.NSSM

# Install service
nssm install VoiceAIAgent "C:\Projects\LMS-VAI-AGENT\venv\Scripts\python.exe" "-m src.main start"
nssm set VoiceAIAgent AppDirectory "C:\Projects\LMS-VAI-AGENT"
nssm set VoiceAIAgent AppStdout "C:\Projects\LMS-VAI-AGENT\logs\service-out.log"
nssm set VoiceAIAgent AppStderr "C:\Projects\LMS-VAI-AGENT\logs\service-err.log"

# Start service
nssm start VoiceAIAgent
```

## Monitoring

### Check Logs
```powershell
# View real-time logs
Get-Content logs\vai-agent.log -Wait -Tail 50
```

### GPU Monitoring
```powershell
# Watch GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

### Performance Metrics
```powershell
# Run benchmark
python scripts/benchmark.py
```

## Troubleshooting

### Audio Not Working
```powershell
# List devices and verify
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
python -c "import sounddevice as sd; import numpy as np; print('Recording...'); audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1); sd.wait(); print('Done')"
```

### GPU Not Used
```powershell
# Verify CUDA available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### Out of Memory
- Reduce Whisper model size (`WHISPER_MODEL=medium`)
- Lower LMStudio context length
- Reduce batch size in LMStudio

## Updates

### Pull Latest Code
```powershell
git pull origin main
pip install -r requirements.txt --upgrade
```

### Backup Configuration
```powershell
# Backup before updates
Copy-Item .env .env.backup
Copy-Item -Recurse models models.backup
```

## Security Notes

1. **Firewall**: Keep LMStudio on localhost only
2. **MCP Server**: Use strong API keys
3. **Audio**: Ensure microphone access is restricted
4. **Logs**: Rotate and encrypt sensitive logs

---

**System Ready**: Your GMKTEC Evo-X2 with 128GB VRAM will run this agent at full performance with all features enabled! 🚀
