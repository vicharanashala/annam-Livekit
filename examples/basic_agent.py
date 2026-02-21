import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import JobContext, JobProcess, Agent, AgentSession, AgentServer, cli, mcp
from livekit.plugins import silero, openai

import voxtral

load_dotenv()

# Fix for Parler TTS Sample Rate
from livekit.plugins.openai import tts as openai_tts
openai_tts.SAMPLE_RATE = 44100
# openai_tts.SAMPLE_RATE = 34000

# Configure logging
logger = logging.getLogger() # Root logger to capture all logs including livekit
logger.setLevel(logging.INFO)

# Configure logging to file
file_handler = logging.FileHandler("/home/aic_u2/Shubhankar/Livekit/github/annam-Livekit/examples/agent.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



logger.info("Agent starting up - logging configured successfully")

MCP_SERVERS = {
    "weather": "http://100.100.108.28:9004/mcp",
    "golden": "http://100.100.108.28:9001/mcp",
    "pop": "http://100.100.108.28:9002/mcp",
    "market": "http://100.100.108.28:9003/mcp",
    "faq-videos": "http://100.100.108.28:9005/mcp"
}

VAD_CONFIG = {
    "activation_threshold": 0.6,    # Stricter than default (0.5)
    "min_speech_duration": 0.2,     # Ignore short pops/clicks
    "min_silence_duration": 0.8,    # Wait 0.8s silence before processing
}

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                """ ROLE & PERSONA
You are **AjraSakha**, an AI assistant for Indian farmers. ALWAYS MENTION SOURCE OF YOUR ANSWER. You sound like a helpful, polite, and practical Agriculture Officer talking on a phone call.
### CONVERSATIONAL STYLE (CRITICAL)
Reply in English or Hindi based on user input.
* **Tone:** Natural, calm, simple, and spoken. Keep sentences relatively short.
* **NO LISTS OR NUMBERING:** Never use bullet points, numbered lists, or say "Point 1", "Point 2". Speak in fluid paragraphs.
* **FLUID TRANSITIONS:** Connect your thoughts using natural transitions like "Also", "Another important thing is", "Besides that", or "Moving on to".
* **NO MARKDOWN:** Do not use bolding, italics, lists, headers, or any other markdown symbols like * ,# etc or any other script apart from Roman or Devanagari in the response.
* **Scope:** Strictly Indian agriculture (crops, soil, pests, schemes, machinery). If unrelated, politely decline.
REMEMBER: ANSWER ONLY BASED ON TOOL RESPONSE 
### EXECUTION WORKFLOW
Do not call more than 5 tools in a single request.
1.  **Context Acquisition:**
    * **Weather:** Ask for **Pincode**. Use `get_current_weather` and dont use get_weather_forecast tool. Always state: *"Weather [forecast] information provided by the India Meteorological Department (IMD)."*
    * **Agriculture:** when user ask Agriculture question You **MUST** obtain the **STATE** and **CROP**  . Ask politely if missing and call the tool as mentioned below before answering agricultural query.
     
2.  **Data Retrieval Hierarchy (Strict Order):**
    First Ask state name if not provided
    * 1st: **Golden Dataset**
    * 2nd: **Package of Practices (PoP)**
    * 3rd: **Reviewer System:** If info is missing in 1 & 2, politely ask for the farmer's phone number & email to log the query.

### TOPIC-SPECIFIC RESPONSE STANDARDS
Flow naturally without numbering.
* **Disease/Pest Management:** Describe symptoms and severity first. For severe cases, suggest 3 legal fungicides/insecticides with different modes of action.
* **Fertilizers:** Emphasize soil testing. Explain organic vs. chemical balance and NPK timing.
* **Varieties:** Mention local practices and safety.

**REMINDER:** Speak like a human on a call."""


# """
# You are a voice-based assistant.
# User input will arrive as audio, and transcripts are not preserved across turns.
# CALL THE TOOL WHEN YOU HAVE ALL INFORMATION
# At the start of every response, briefly restate in one short sentence what you understood from the user’s last message. This is only to confirm context and must be very concise, after this call the tool if needed answer the question based on retrived information only.

# Weather Queries

# If the user asks anything related to weather:
# Ask for the user’s pincode if it is not already provided.
# Use only the get_current_weather tool to fetch weather information.
# Do not use any other weather-related tool.

# Agricultural Queries

# If the user asks an agricultural question:
# You must identify the State and Crop.
# Only these two details are required to proceed.
# First, call get_context_from_golden_dataset using the user’s question, State, and Crop.
# If the retrieved content is not relevant or no suitable answer is found, then call get_context_from_package_of_practices.
# Do not use any other agricultural tools.

# Response Rules

# Speak naturally, as if talking to the user.
# Do not use symbols, bullet points, markdown, or special formatting.
# Use plain conversational text suitable for text to speech.
# Do not invent or add information beyond what is returned by tools.
# If no relevant data is found, clearly say that verified information is not available.

# Tool Usage Output
# Do not call multiple tools unless explicitly allowed by the rules above.
# """
            ),
        )

    async def on_enter(self, **kwargs):
        await self.session.generate_reply()

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(**VAD_CONFIG)

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Initialize Voxtral Wrapper
    v_state = voxtral.VoxtralState()

    # Initialize Voxtral STT (Audio Capture)
    whisper_stt = voxtral.VoxtralSTT(
        state=v_state
    )

    # Initialize Voxtral LLM (API Caller)
    # Using the port/model from the user's bash script
    BASE_URL = "http://100.100.108.28:8007/v1"
    MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    
    ollama_llm = voxtral.VoxtralLLM(
        base_url=BASE_URL,
        model=MODEL_ID,
        api_key="voxtral", 
        state=v_state
    )
    
    # Initialize Kokoro TTS using OpenAI Plugin
    # kokoro_tts = openai.TTS(
    #     base_url="http://localhost:8880/v1",
    #     model="kokoro",
    #     # voice="af_sky",
    #     voice ="hf_alpha",
    #     api_key="kokoro", )

    # )
    kokoro_tts = openai.TTS(
    base_url="http://localhost:8003/v1",
    model="ai4bharat/IndicF5",     # must match what your server routes/accepts
    api_key="local-anything",      # usually required by the plugin, but your server can ignore it
    response_format="pcm",         # IMPORTANT: plugin expects pcm bytes
)
  
    
    # kokoro_tts = openai.TTS(
    #         base_url="http://localhost:8000/v1",
    #         model="ai4bharat/indic-parler-tts",
    #         # model="parler-tts/parler-tts-mini",

    #         voice="Feminine",  # or any other voice
    #         api_key="indic-parler",
    #         # response_format="pcm",  # Indic Parler TTS returns raw PCM for streaming
    #     )

    # Convert MCP_SERVERS dict to list of MCPServerHTTP
    mcp_servers_list = [mcp.MCPServerHTTP(url=url) for url in MCP_SERVERS.values()]
    
    # Ensure VAD is loaded (fallback if prewarm didn't run)
    vad_instance = ctx.proc.userdata.get("vad")
    if not vad_instance:
        logger.info("VAD not found in userdata, loading now")
        vad_instance = silero.VAD.load(**VAD_CONFIG)

    session = AgentSession(
        vad=vad_instance,
        stt=whisper_stt,
        llm=ollama_llm,
        tts=kokoro_tts,
        mcp_servers=mcp_servers_list,
        max_tool_steps=15,
    )

    await session.start(agent=MyAgent(), room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
