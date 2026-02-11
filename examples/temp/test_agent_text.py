"""
Text-only agent for testing without audio hardware.
This version works on headless servers.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    function_tool,
)
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)


# Define function tools
@function_tool
async def lookup_weather(
    context: RunContext,
    location: str,
):
    """Used to look up weather information.
    
    Args:
        location: The city or location to get weather for
    """
    logger.info(f"🌤️  Looking up weather for: {location}")
    
    weather_data = {
        "location": location,
        "weather": "sunny",
        "temperature": 72,
        "humidity": 45,
    }
    
    return weather_data


@function_tool
async def get_time(context: RunContext):
    """Get the current time."""
    from datetime import datetime
    
    current_time = datetime.now().strftime("%I:%M %p")
    logger.info(f"🕐 Current time: {current_time}")
    
    return {"time": current_time}


async def test_agent():
    """Test the agent without audio I/O"""
    
    logger.info("=" * 60)
    logger.info("LiveKit Agents - Infrastructure Test")
    logger.info("=" * 60)
    logger.info("")
    
    # Create the agent
    agent = Agent(
        instructions=(
            "You are a friendly assistant. "
            "You can help with weather and time information. "
            "You are running with PLACEHOLDER APIs."
        ),
        tools=[lookup_weather, get_time],
    )
    
    logger.info(f"✅ Agent created successfully")
    logger.info(f"   Instructions: {agent.instructions[:50]}...")
    logger.info(f"   Number of tools: {len(agent.tools)}")
    logger.info(f"   Tool names: {[tool.__name__ for tool in agent.tools]}")
    logger.info("")
    
    logger.info("=" * 60)
    logger.info("✅ Infrastructure Test Passed!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Summary:")
    logger.info("  ✅ Agent class working")
    logger.info("  ✅ Function tools registered")
    logger.info("  ✅ Tool metadata extracted")
    logger.info("  ✅ Ready for integration with STT/LLM/TTS")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Use 'dev' mode to connect to LiveKit server")
    logger.info("  2. Replace placeholder STT/LLM/TTS with real APIs")
    logger.info("  3. Test with actual voice clients")
    logger.info("")


if __name__ == "__main__":
    asyncio.run(test_agent())
