import logging
from dotenv import load_dotenv
from livekit.agents import Agent, JobContext, JobProcess, AgentSession, AgentServer, cli, mcp
from livekit.plugins import silero, openai

# Load environment variables from .env file
load_dotenv()


# =====================================================
# LOGGING CONFIG
# =====================================================

# Root logger to capture all logs including livekit
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure logging to file
#file_handler = logging.FileHandler("/home/aic_u2/Shubhankar/Livekit/github/annam-Livekit/Livekit/log")
file_handler = logging.FileHandler("./log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Also add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Agent starting up - logging configured successfully")


# =====================================================
# MCP SERVER URLS
# =====================================================

MCP_SERVERS = {
    "weather": "http://100.100.108.28:9004/mcp",
    "golden": "http://100.100.108.28:9001/mcp",
    "pop": "http://100.100.108.28:9002/mcp",
    "faq_videos": "http://100.100.108.28:9005/mcp",
}


# =====================================================
# VAD CONFIG
# =====================================================

VAD_CONFIG = {
    "activation_threshold": 0.6,    # Stricter than default (0.5)
    "min_speech_duration": 0.2,     # Ignore short pops/clicks
    "min_silence_duration": 0.8,    # Wait 0.8s silence before processing
}


# =====================================================
# AGENT CLASS
# =====================================================

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an intelligent voice assistant.
            Use tools when relevant:
            - weather for weather info
            - golden for golden dataset Q&A
            - pop for recommendation
            - faq_videos for video suggestions
            """,
        )

    async def on_enter(self, **kwargs):
        await self.session.generate_reply()


# =====================================================
# SERVER INIT
# =====================================================

server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm function to load VAD before session starts"""
    proc.userdata["vad"] = silero.VAD.load(**VAD_CONFIG)


server.setup_fnc = prewarm


# =====================================================
# RTC SESSION ENTRYPOINT
# =====================================================

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Connecting to room: {ctx.room.name}")

    # ---------------------------
    # STT - Whisper
    # ---------------------------
    whisper_stt = openai.STT(
        base_url="http://localhost:8030/v1",
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        api_key="whisper",  # required but ignored,
        language="hi"
    )
    logger.info("Whisper STT initialized")

    # ---------------------------
    # LLM - Ollama (OpenAI compatible)
    # ---------------------------
    ollama_llm = openai.LLM(
        base_url="http://100.100.108.100:8081/v1",
        model="Qwen/Qwen3-30B-A3B",
        api_key="ollama",  # required but ignored
    )
    logger.info("Ollama LLM initialized")

    # ---------------------------
    # TTS - Kokoro
    # ---------------------------
    kokoro_tts = openai.TTS(
        base_url="http://localhost:8003/v1",
        model="kokoro",
        api_key="kokoro",
        response_format="pcm"

    )
    logger.info("Kokoro TTS initialized")

    # ---------------------------
    # CONVERT MCP SERVERS TO LIST
    # ---------------------------
    mcp_servers_list = [mcp.MCPServerHTTP(url=url) for url in MCP_SERVERS.values()]
    logger.info(f"Initialized {len(mcp_servers_list)} MCP servers")

    # ---------------------------
    # ENSURE VAD IS LOADED
    # ---------------------------
    vad_instance = ctx.proc.userdata.get("vad")
    if not vad_instance:
        logger.info("VAD not found in userdata, loading now")
        vad_instance = silero.VAD.load(**VAD_CONFIG)

    # ---------------------------
    # CREATE AGENT SESSION
    # ---------------------------
    session = AgentSession(
        vad=vad_instance,
        stt=whisper_stt,
        llm=ollama_llm,
        tts=kokoro_tts,
        mcp_servers=mcp_servers_list,
    )

    logger.info("Agent session created successfully")

    await session.start(agent=MyAgent(), room=ctx.room)
    await ctx.connect()

    logger.info("Agent started and ready")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    cli.run_app(server)
