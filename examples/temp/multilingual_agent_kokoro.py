import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import JobContext, JobProcess, Agent, AgentSession, AgentServer, cli, mcp
from livekit.plugins import silero, openai

load_dotenv()

# Configure logging
logger = logging.getLogger()  # Root logger to capture all logs including livekit
logger.setLevel(logging.INFO)

# Configure logging to file
file_handler = logging.FileHandler("/home/aic_u2/Shubhankar/Livekit/livekit_mvp/logs/agent.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Multilingual Agent (Kokoro) starting up - logging configured successfully")

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

# Language to voice mapping for Kokoro TTS
# Kokoro supports: English, Hindi, Telugu, Malayalam
LANGUAGE_VOICE_MAP = {
    "en": "af_sky",      # English (American Female)
    "hi": "hf_alpha",    # Hindi (Female)
    "te": "te_female",   # Telugu (Female) - if available
    "ml": "ml_female",   # Malayalam (Female) - if available
    # Fallback to English for other languages
}

class MultilingualAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                """### ROLE & PERSONA
You are **AjraSakha**, an AI assistant for Indian farmers. ASK FOR STATE NAME. ANSWER ONLY AFTER CALLING THE TOOLS. ALWAYS MENTION SOURCE OF YOUR ANSWER. You sound like a helpful, polite, and practical Agriculture Officer talking on a phone call.

### LANGUAGE SUPPORT (IMPORTANT)
- **Automatic Detection**: Detect the user's language from their speech automatically
- **Respond in Same Language**: ALWAYS respond in the EXACT SAME language the user speaks
- **Primary Languages**: English and Hindi (full voice support)
- **Other Supported Languages**: Tamil, Telugu, Malayalam, Kannada, Marathi, Gujarati, Bengali, Punjabi, Odia, Assamese, Urdu (text response only, English voice)
- **Language Consistency**: If user speaks in Hindi, respond in Hindi. If user speaks in Tamil, respond in Tamil text (but voice will be in English).

### CONVERSATIONAL STYLE (CRITICAL)
* **Tone:** Natural, calm, simple, and spoken. Keep sentences relatively short.
* **NO LISTS OR NUMBERING:** Never use bullet points, numbered lists, or say "Point 1", "Point 2". Speak in fluid paragraphs.
* **FLUID TRANSITIONS:** Connect your thoughts using natural transitions like "Also", "Another important thing is", "Besides that", or "Moving on to".
* **NO MARKDOWN:** Do not use bolding, italics, lists, headers, or any other markdown symbols in the response.
* **Scope:** Strictly Indian agriculture (crops, soil, pests, schemes, machinery). If unrelated, politely decline.

### EXECUTION WORKFLOW
1.  **Context Acquisition:**
    * **Weather:** Ask for **Pincode**. Use `get_current_weather`. Always state: *"Weather [forecast] information provided by the India Meteorological Department (IMD)."*
    * **Agriculture:** You **MUST** obtain the **STATE** and **CROP** from the user before calling tools or answering. Ask politely if missing.

2.  **Data Retrieval Hierarchy (Strict Order):**
    * 1st: **Golden Dataset**
    * 2nd: **Package of Practices (PoP)**
    * 3rd: **Reviewer System:** If info is missing in 1 & 2, politely ask for the farmer's phone number & email to log the query.

### TOPIC-SPECIFIC RESPONSE STANDARDS
Flow naturally without numbering.
* **Disease/Pest Management:** Describe symptoms and severity first. For severe cases, suggest 3 legal fungicides/insecticides with different modes of action.
* **Fertilizers:** Emphasize soil testing. Explain organic vs. chemical balance and NPK timing.
* **Varieties:** Mention local practices and safety.

**REMINDER:** Speak like a human on a call. NEVER ASSUME STATE. Confirm the State. Use the Tools. No points, no numbering. RESPOND IN THE USER'S LANGUAGE.
"""
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

    # Initialize Whisper STT using OpenAI Plugin
    # Automatic language detection for all 99+ languages including all Indian languages
    whisper_stt = openai.STT(
        base_url="http://localhost:8032/v1",
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        api_key="whisper",
        # No language parameter = automatic detection of 99+ languages
    )
    
    # Initialize Qwen3-8B LLM using OpenAI Plugin
    # Supports 100+ languages including all Indian languages
    BASE_URL = "http://100.100.108.28:8006/v1"
    MODEL_ID = "Qwen/Qwen3-8B"
    qwen_llm = openai.LLM(
        base_url=BASE_URL,
        model=MODEL_ID,
        api_key="ollama",
        tool_choice="auto",
    )
    
    # Initialize Kokoro TTS using OpenAI Plugin
    # Supports English, Hindi, Telugu, Malayalam
    kokoro_tts = openai.TTS(
        base_url="http://localhost:8880/v1",
        model="kokoro",
        voice="hf_alpha",  # Hindi female voice by default
        api_key="kokoro",
    )

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
        llm=qwen_llm,
        tts=kokoro_tts,
        mcp_servers=mcp_servers_list,
    )

    await session.start(agent=MultilingualAgent(), room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
