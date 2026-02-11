import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import JobContext, JobProcess, Agent, AgentSession, AgentServer, cli, mcp
from livekit.plugins import openai

load_dotenv()

# Configure logging
logger = logging.getLogger() # Root logger to capture all logs including livekit
logger.setLevel(logging.INFO)

# Configure logging to file
file_handler = logging.FileHandler("/home/aic_u2/Shubhankar/Livekit/livekit_mvp/logs/agent.log")
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



class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                """### ROLE & PERSONA
You are **AjraSakha**, an AI assistant for Indian farmers. ASK FOR STATE NAME. ANSWER ONLY AFTER CALLING THE TOOLS. You sound like a helpful, polite, and practical Agriculture Officer talking on a phone call.

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

**REMINDER:** Speak like a human on a call. NEVER ASSUME STATE. Confirm the State. Use the Tools. No points, no numbering.
"""
            ),
        )

    async def on_enter(self, **kwargs):
        await self.session.generate_reply()

server = AgentServer()

# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load(**VAD_CONFIG)

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Initialize Realtime Model (Qwen Omni)
    realtime_model = openai.realtime.RealtimeModel(
        base_url="http://100.100.108.28:8901/v1",
        model="Qwen/Qwen3-8B",
        api_key="dummy",
    )
    
    # Convert MCP_SERVERS dict to list of MCPServerHTTP
    mcp_servers_list = [mcp.MCPServerHTTP(url=url) for url in MCP_SERVERS.values()]
    
    session = AgentSession(
        llm=realtime_model,
        mcp_servers=mcp_servers_list,
    )

    await session.start(agent=MyAgent(), room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
