#!/bin/bash
set -e

echo "Starting LMS Voice AI Agent..."

# Check if LMStudio is accessible
if ! curl -s -f "$LMSTUDIO_BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "WARNING: LMStudio is not accessible at $LMSTUDIO_BASE_URL"
    echo "Please ensure LMStudio is running on the host machine."
fi

# Check if MCP server is accessible
if ! curl -s -f "$MCP_SERVER_URL/health" > /dev/null 2>&1; then
    echo "WARNING: MCP server is not accessible at $MCP_SERVER_URL"
    echo "Agent will start but tool execution may fail."
fi

# Download models if needed
if [ "$WHISPER_MODEL" != "" ]; then
    echo "Ensuring Whisper model '$WHISPER_MODEL' is available..."
    python -c "import whisper; whisper.load_model('$WHISPER_MODEL', download_root='/app/models')" || echo "Whisper model download deferred to runtime"
fi

if [ "$TTS_MODEL" != "" ]; then
    echo "Ensuring TTS model '$TTS_MODEL' is available..."
    python -c "from TTS.api import TTS; TTS('$TTS_MODEL')" || echo "TTS model download deferred to runtime"
fi

echo "Configuration:"
echo "  LMStudio: $LMSTUDIO_BASE_URL"
echo "  MCP Server: $MCP_SERVER_URL"
echo "  Whisper Model: $WHISPER_MODEL"
echo "  TTS Device: $TTS_DEVICE"
echo "  Debug: $DEBUG"

# Execute main command
exec "$@"
