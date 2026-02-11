# Voxtral Module Documentation

## Overview
The `voxtral.py` module implements a custom integration for a multimodal Large Language Model (Voxtral) within the LiveKit ecosystem. Unlike traditional pipelines that separate Speech-to-Text (STT) and LLM processing, this module treats audio as a direct input modality for the LLM.

## Architecture
The module implements a "fake" STT layer that essentially acts as an audio capture and encoding mechanism. The captured audio is not transcribed locally but is instead encoded into Base64 and injected directly into the LLM's context window.

### Data Flow
1. **Audio Capture**: `VoxtralSTT` receives audio frames from the LiveKit room.
2. **Buffering & Encoding**: Audio is buffered and encoded to Base64 WAV format.
3. **Signal Event**: `VoxtralSTT` emits a special `[AUDIO_INPUT]` transcript event.
4. **LLM Injection**: `VoxtralLLM` detects the `[AUDIO_INPUT]` placeholder and replaces it with the actual Base64 audio data in compliance with OpenAI-compatible multimodal APIs.
5. **Response**: The LLM processes the audio directly and generates a text response (or tool calls).

## Components

### 1. `VoxtralState`
A shared state container used to pass audio data between the STT and LLM components.
- **`current_audio_b64`**: Stores the latest utterance as a Base64 string.

### 2. `VoxtralSTT`
Inherits from `livekit.agents.stt.STT`.
- **Purpose**: Captures audio, buffers it, and triggers the LLM.
- **Behavior**:
  - Does **not** perform actual transcription.
  - Returns a fixed transcript `[AUDIO_INPUT]` to signal the Agent framework that input is ready.
  - Updates `VoxtralState` with the actual audio data.

### 3. `VoxtralSpeechStream`
Inherits from `livekit.agents.stt.SpeechStream`.
- **Purpose**: Handles the buffering of audio frames.
- **Key Methods**:
  - `push_frame`: Accumulates audio data.
  - `flush`: Encodes the accumulated audio to Base64, updates the state, and emits the `[AUDIO_INPUT]` event.

### 4. `VoxtralLLM`
Inherits from `livekit.agents.llm.LLM`.
- **Purpose**: Prepares and sends requests to an OpenAI-compatible API (e.g., vLLM or Mistral) with audio support.
- **Key Features**:
  - **Audio Injection**: Detects `[AUDIO_INPUT]` messages and constructs an `input_audio` payload.
  - **System Instruction Handling**: Automatically extracts system instructions and prepends them to the multimodal message if audio is present.
  - **Sanitization**: Ensures message ordering complies with strict API requirements (e.g., ensuring `Tool` messages are followed by `Assistant` messages).

### 5. `VoxtralLLMStream`
Inherits from `livekit.agents.llm.LLMStream`.
- **Purpose**: Handles the streaming response from the API.
- **Tool Calling Support**:
  - **Native**: Supports standard OpenAI tool calling structures.
  - **Text-Based**: Fallback support for parsing JSON-formatted tool calls embedded in text (e.g., `[TOOL_CALLS][{"name": "...", "arguments": ...}]`).

## Usage Example

```python
from livekit.agents import PipelineAgent
from .voxtral import VoxtralSTT, VoxtralLLM, VoxtralState

# 1. Initialize Shared State
state = VoxtralState()

# 2. Configure Components with Shared State
stt = VoxtralSTT(state=state)
llm = VoxtralLLM(
    base_url="http://localhost:8000/v1",
    model="voxtral-model-v1",
    api_key="secret",
    state=state
)

# 3. Create Agent
agent = PipelineAgent(
    stt=stt,
    llm=llm,
    # ... other components
)
```

## Known Limitations / Quirks
- **Audio Dependency**: The LLM context injection relies strictly on the `[AUDIO_INPUT]` marker.
- **Message Ordering**: Contains specific logic to fix message ordering for models like Mistral/vLLM that are strict about the `User` -> `Assistant` -> `Tool` flow.
- **Text-Based Tool Calls**: Includes a robust parser for text-based tool calls to handle models that might not support native tool calling perfectly or "leak" tool calls into the text stream.
