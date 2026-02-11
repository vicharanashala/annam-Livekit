
from livekit.agents.llm import ChatContext, ChatMessage
import asyncio

async def main():
    ctx = ChatContext()
    # Create a message with string content
    # Create a message with list content
    msg = ChatMessage(role="user", content=["[AUDIO_INPUT]"])
    # msg = ChatMessage(role="user", content=["[AUDIO_INPUT]"]) # We can just add directly
    # ctx.add_message(msg) # API might behave differently
    
    # Correct usage based on SDK docs (or implied)
    msg = ChatMessage(role="user", content=["[AUDIO_INPUT]"])
    # If add_message expects a message object
    ctx.messages.append(msg) if hasattr(ctx, "messages") else None
    # Wait, earlier inspecting ChatContext showed no messages attr, but _items
    # Let's try appending to _items directly if public API is unclear, OR use add_message
    
    # Directly append to _items as verified in inspection
    if hasattr(ctx, "_items"):
        ctx._items.append(msg)
    else:
        print("Error: ctx has no _items")
        return

    print(f"Original content type: {type(msg.content)}")
    print(f"Original content: {msg.content}")
    
    messages_before, _ = ctx.to_provider_format("openai")
    print(f"OpenAI format before: {messages_before}")
    
    # Simulate the update
    transcription = "My paddy have leaf blight."
    msg.content = [transcription] # Assigning as list
    
    print(f"Updated content type: {type(msg.content)}")
    print(f"Updated content: {msg.content}")
    
    messages_after, _ = ctx.to_provider_format("openai")
    print(f"OpenAI format after: {messages_after}")

    # Inspect if any weird iteration happens
    if isinstance(msg.content, str):
         print("Content is string. Iterating it gives chars:")
         print(list(msg.content))

if __name__ == "__main__":
    asyncio.run(main())
