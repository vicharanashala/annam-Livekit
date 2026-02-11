"""
Basic Voice Agent Example using LiveKit Agents Framework

This example demonstrates:
- Using the official LiveKit Agents framework
- Placeholder STT, LLM, TTS (no API keys needed)
- Function tools
- Agent session management

To run:
    python examples/basic_agent.py console  # Terminal mode
    python examples/basic_agent.py dev      # Development mode
    python examples/basic_agent.py start    # Production mode
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import silero
from loguru import logger
from dotenv import load_dotenv

# Import custom plugins

from livekit.plugins import openai

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
    logger.info(f"Looking up weather for: {location}")
    
    # Placeholder weather data
    weather_data = {
        "location": location,
        "weather": "sunny",
        "temperature": 72,
        "humidity": 45,
        "wind_speed": 10,
    }
    
    return weather_data


@function_tool
async def get_time(context: RunContext):
    """Get the current time."""
    from datetime import datetime
    
    current_time = datetime.now().strftime("%I:%M %p")
    logger.info(f"Current time: {current_time}")
    
    return {"time": current_time}


# Main entrypoint
async def entrypoint(ctx: JobContext):
    """
    Entry point for the agent.
    This is called when a new session starts.
    """
    logger.info("Agent entrypoint called")
    
    # Connect to the room
    await ctx.connect()
    logger.info("Connected to room")
    
    # Create the agent with instructions and tools
    agent = Agent(
        instructions=(
            "You are a friendly voice assistant built by LiveKit. "
            "You can help users with weather information and tell them the time. "
            "Be conversational and helpful. "
            "NOTE: You are currently running with PLACEHOLDER APIs for STT, LLM, and TTS. "
            "This is just for testing the infrastructure."
        ),
        tools=[lookup_weather, get_time],
    )
    
    logger.info(f"Agent created with {len(agent.tools)} tools")
    
    # Create agent session with self-hosted Whisper STT, Ollama LLM, and Edge TTS
    
    # Initialize Whisper STT using OpenAI Plugin
    # Note: connect to HTTP endpoint, not WS
    whisper_stt = openai.STT(
        base_url="http://localhost:8030/v1",
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        api_key="whisper", # Required but ignored
    )
    
    # Initialize Ollama LLM using OpenAI Plugin
    ollama_llm = openai.LLM(
        base_url="http://100.100.108.27:11434/v1",
        model="nemotron-mini:4b",
        api_key="ollama", # Required but ignored
    )
    
    # Initialize Kokoro TTS using OpenAI Plugin
    kokoro_tts = openai.TTS(
        base_url="http://localhost:8880/v1",
        model="kokoro",
        voice="af_bella",
        api_key="kokoro", # Required but ignored
    )
    
    session = AgentSession(
        vad=silero.VAD.load(),  # Voice Activity Detection (real)
        stt=whisper_stt,        # Self-hosted Whisper on port 8030 ✅
        llm=ollama_llm,         # Ollama via OpenAI Plugin ✅
        tts=kokoro_tts,         # Kokoro TTS via OpenAI Plugin ✅
    )
    
    logger.info("Agent session created with Whisper STT, Ollama LLM, and Kokoro TTS (via OpenAI)")
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    logger.info("Agent session started")
    
    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and let them know you can help with weather and time. Mention that you're running in placeholder mode."
    )
    
    logger.info("Initial greeting generated")


# Run the application
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LiveKit Agents MVP - Basic Agent")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This agent uses PLACEHOLDER APIs (no keys needed)")
    logger.info("To use real APIs, update the session configuration")
    logger.info("")
    

    
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
    ))
