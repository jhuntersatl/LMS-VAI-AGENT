"""
Simple test of core components without audio.
"""

import asyncio
from src.config import get_config
from src.lm_client import LMStudioClient

async def test_basic():
    print("🧪 Testing LMS Voice AI Agent (without audio)")
    print("=" * 60)
    
    config = get_config()
    
    # Test LMStudio
    print("\n1. Testing LMStudio connection...")
    lm = LMStudioClient(config.lmstudio)
    
    if await lm.health_check():
        print("   ✓ LMStudio is available")
        
        # Test generation
        response = await lm.generate("What is 2+2?", max_tokens=50)
        print(f"   ✓ Response: {response}")
        
        # Test intent parsing
        intent = await lm.parse_intent("What's the weather today?")
        print(f"   ✓ Intent parsed: {intent}")
    else:
        print("   ✗ LMStudio not available")
        return
    
    print("\n✅ Core functionality working!")
    print("\n📝 Note: Audio components need ARM64-compatible libraries")
    print("   Whisper and TTS will work once audio I/O is fixed")

if __name__ == "__main__":
    asyncio.run(test_basic())
