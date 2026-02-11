"""
Test script to verify LiveKit Agents installation
This doesn't require audio devices or PortAudio
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 60)
print("LiveKit Agents Installation Test")
print("=" * 60)
print()

# Test 1: Import LiveKit
try:
    import livekit
    print("✅ LiveKit imported successfully")
except Exception as e:
    print(f"❌ Failed to import LiveKit: {e}")
    sys.exit(1)

# Test 2: Import LiveKit Agents
try:
    from livekit import agents
    print("✅ LiveKit Agents imported successfully")
except Exception as e:
    print(f"❌ Failed to import LiveKit Agents: {e}")
    sys.exit(1)

# Test 3: Import core components
try:
    from livekit.agents import (
        Agent,
        AgentSession,
        JobContext,
        RunContext,
        WorkerOptions,
        cli,
        function_tool,
    )
    print("✅ Core components imported successfully")
except Exception as e:
    print(f"❌ Failed to import core components: {e}")
    sys.exit(1)

# Test 4: Import Silero VAD
try:
    from livekit.plugins import silero
    print("✅ Silero VAD plugin imported successfully")
except Exception as e:
    print(f"❌ Failed to import Silero: {e}")
    sys.exit(1)

# Test 5: Import placeholder plugins
try:
    from plugins import (
        PlaceholderSTT,
        PlaceholderLLM,
        PlaceholderTTS,
    )
    print("✅ Placeholder plugins imported successfully")
except Exception as e:
    print(f"❌ Failed to import placeholder plugins: {e}")
    sys.exit(1)

# Test 6: Create a simple agent
try:
    @function_tool
    async def test_tool(context: RunContext):
        """Test tool"""
        return {"status": "ok"}
    
    agent = Agent(
        instructions="Test agent",
        tools=[test_tool],
    )
    print("✅ Agent created successfully")
    print(f"   Agent has {len(agent.tools)} tool(s)")
except Exception as e:
    print(f"❌ Failed to create agent: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("🎉 All tests passed! Installation is working correctly.")
print("=" * 60)
print()
print("Note: Console mode requires PortAudio (audio library)")
print("      Use 'dev' mode instead for server environments:")
print()
print("      python examples/basic_agent.py dev")
print()
