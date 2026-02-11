
import asyncio
import logging
from livekit.agents import llm, APIConnectOptions, stt
from livekit.agents.llm import ToolContext, tool_context
import voxtral

# Mock tool
@llm.function_tool(description="Get the current weather")
def get_weather(location: str):
    print(f"Getting weather for {location}")
    return "Sunny"

async def main():
    print("Testing Voxtral Integration...")
    
    # 1. Test VoxtralState
    state = voxtral.VoxtralState()
    print("VoxtralState created.")

    # 2. Test VoxtralSTT
    stt_inst = voxtral.VoxtralSTT(state=state)
    print("VoxtralSTT created.")

    # 3. Test ToolContext Creation
    # FunctionTool
    tools = [get_weather]
    tc = ToolContext(tools)
    print("ToolContext created with 1 tool.")

    # 4. Test VoxtralLLM initialization
    v_llm = voxtral.VoxtralLLM(
        base_url="http://fake:1234",
        model="fake-model",
        api_key="fake",
        state=state
    )
    print("VoxtralLLM created.")

    # Test STT manually
    event = await stt_inst._recognize_impl(buffer=[])
    print(f"STT Event: {event}")
    if event.alternatives[0].language == "":
        print("STT Language set correctly.")
    else:
        print("STT Language incorrect.")


    # 5. Test Chat Call (Simulated)
    # We call chat() just to see if it processes tools correctly without crashing
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")
    
    # Test passing tools as keyword arg
    tools_list = [get_weather]
    stream = v_llm.chat(chat_ctx=chat_ctx, tools=tools_list, conn_options=APIConnectOptions())
    print("VoxtralLLM.chat(tools=..., conn_options=...) called successfully. Iterating stream...")
    
    # Iterate over stream to catch generation errors
    async for chunk in stream:
        print(f"Received chunk: {chunk}")
        
    print("Stream finished.")

    # Check if tools were extracted
    if len(v_llm._tools) == 1:
        print("Success: 1 tool extracted.")
        print(f"Tool: {v_llm._tools[0]}")
    else:
        print(f"Failure: Expected 1 tool, found {len(v_llm._tools)}")
        exit(1)

    print("Verification Passed!")

if __name__ == "__main__":
    asyncio.run(main())
