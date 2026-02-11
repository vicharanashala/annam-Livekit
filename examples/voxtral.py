import asyncio
import base64
import json
import logging
import io
import subprocess
import wave
import uuid
from typing import List, Optional, Union, AsyncIterable, Dict, Any

import aiohttp
from livekit.agents import stt, llm, tokenize, utils, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.llm import utils as llm_utils
from livekit.agents.llm import tool_context
from livekit.rtc import AudioFrame

logger = logging.getLogger("voxtral")

class VoxtralState:
    def __init__(self):
        self.current_audio_b64: Optional[str] = None

class VoxtralSTT(stt.STT):
    def __init__(
        self,
        *,
        state: VoxtralState,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self.state = state

    async def _recognize_impl(
        self,
        buffer: AudioFrame | List[AudioFrame],
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        logger.info(f"VoxtralSTT._recognize_impl called with {len(frames)} frames")
        
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="", confidence=0.0, language=language or "")]
            )

        # Merge audio data
        wav_buffer = io.BytesIO()
        sample_rate = frames[0].sample_rate
        channels = frames[0].num_channels
        
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            for frame in frames:
                wf.writeframes(frame.data)
                
        wav_b64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')
        self.state.current_audio_b64 = wav_b64
        logger.info(f"VoxtralSTT state updated with audio. Length: {len(wav_b64)}")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="[AUDIO_INPUT]", confidence=1.0, language=language or "")]
        )

    def stream(self) -> stt.SpeechStream:
        print("DEBUG: VoxtralSTT.stream() called")
        logger.info("VoxtralSTT.stream() called")
        return VoxtralSpeechStream(self)

class VoxtralSpeechStream(stt.SpeechStream):
    def __init__(self, stt: VoxtralSTT):
        super().__init__(stt=stt)
        self._stt = stt
        self._wav_buffer = io.BytesIO()
        self._sample_rate: Optional[int] = None
        self._channels: Optional[int] = None
        self._event_queue: asyncio.Queue[stt.SpeechEvent] = asyncio.Queue()
        self._frame_count = 0
        print("DEBUG: VoxtralSpeechStream initialized")

    async def _encode_wav(self, pcm_data: bytes, sample_rate: int, channels: int) -> str:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(wav_buffer.getvalue()).decode('utf-8')

    def push_frame(self, frame: AudioFrame):
        self._sample_rate = frame.sample_rate
        self._channels = frame.num_channels
        self._wav_buffer.write(frame.data)
        self._frame_count += 1
        if self._frame_count % 50 == 0:
             print(f"DEBUG: VoxtralSTT received frame {self._frame_count}")

    async def aclose(self):
        pass

    async def flush(self):
        pcm_data = self._wav_buffer.getvalue()
        print(f"DEBUG: VoxtralSTT flushing. PCM data size: {len(pcm_data)} bytes")
        logger.info(f"VoxtralSTT flushing. PCM data size: {len(pcm_data)} bytes")
        if not pcm_data:
            print("DEBUG: VoxtralSTT flush called with empty buffer!")
            logger.warning("VoxtralSTT flush called with empty buffer!")
            return

        self._stt.state.current_audio_b64 = None

        try:
            wav_b64 = await self._encode_wav(pcm_data, self._sample_rate, self._channels)
            self._stt.state.current_audio_b64 = wav_b64
            logger.info(f"VoxtralSTT audio encoded. Base64 length: {len(wav_b64)}")
            
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="[AUDIO_INPUT]", confidence=1.0, language="")]
            )
            self._event_queue.put_nowait(event)

        except Exception as e:
            logger.error(f"Voxtral STT failed: {e}")
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="Error processing audio.", confidence=0.0, language="")]
            )
            self._event_queue.put_nowait(event)

    async def __anext__(self) -> stt.SpeechEvent:
        return await self._event_queue.get()

class VoxtralLLM(llm.LLM):
    def __init__(self, *, base_url: str, model: str, api_key: str, state: VoxtralState):
        super().__init__()
        self.base_url = base_url
        self._model = model
        self.api_key = api_key
        self.state = state
        self._tools = []

    @property
    def model(self) -> str:
        return self._model

    async def _run(self) -> None:
        pass

    def chat(
        self,
        chat_ctx: llm.ChatContext,
        tools: Optional[List[Union[llm.FunctionTool, llm.RawFunctionTool]]] = None,
        fnc_ctx: Optional[llm.ToolContext] = None,
        conn_options: Optional[APIConnectOptions] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        tool_choice: Union[llm.ToolChoice, None] = None,
    ) -> "VoxtralLLMStream":
        self._tools = []
        
        # Use tools list if provided, otherwise fallback to fnc_ctx
        tools_list = tools if tools else (list(fnc_ctx.function_tools.values()) if fnc_ctx else [])

        for tool in tools_list:
            if tool_context.is_raw_function_tool(tool):
                info = tool_context.get_raw_function_info(tool)
                self._tools.append({
                    "type": "function",
                    "function": info.raw_schema
                })
            elif tool_context.is_function_tool(tool):
                # Use utility to build strict schema
                self._tools.append(llm_utils.build_strict_openai_schema(tool))

        return VoxtralLLMStream(self, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, tools=tools_list)

class VoxtralLLMStream(llm.LLMStream):
    def __init__(self, llm_inst: VoxtralLLM, chat_ctx: llm.ChatContext, fnc_ctx: Optional[llm.ToolContext], tools: List[Union[llm.FunctionTool, llm.RawFunctionTool]]):
        super().__init__(llm=llm_inst, chat_ctx=chat_ctx, tools=tools, conn_options=APIConnectOptions())
        self._llm = llm_inst
        self._ctx = chat_ctx
        
    async def _transcribe_audio(self, audio_b64: str, target_message: Any):
        """
        Transcribes the audio and updates the target message in the conversation history.
        """
        try:
            # Prepare payload for transcription
            payload = {
                "model": self._llm.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe this audio in English or Hindi only."},
                            {"type": "input_audio", "input_audio": { "data": audio_b64, "format": "wav" } }
                        ]
                    }
                ]
            }
            
            # Determine API URL - use existing logic
            # Assuming standard OpenAI style endpoint structure
            url = f"{self._llm.base_url}/chat/completions"
            
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                # Propagate API key if available
                if self._llm.api_key:
                     headers["Authorization"] = f"Bearer {self._llm.api_key}"

                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Transcription request failed with status {resp.status}")
                        return
                        
                    resp_data = await resp.json()
                    
                    if "choices" in resp_data and len(resp_data["choices"]) > 0:
                        transcription = resp_data["choices"][0]["message"].get("content", "")
                        
                        if transcription:
                             # Update the message content
                             # target_message is a ChatMessage object
                             # ChatMessage content expects a list, otherwise serialization iterates over the string
                             target_message.content = [transcription]
                             logger.info(f"Transcription successful and history updated: {transcription[:40]}...")
                        else:
                             logger.warning("Transcription returned empty text.")
                    else:
                        logger.warning("Transcription response had no choices.")

        except Exception as e:
            logger.error(f"Transcription failed: {e}")

    async def _run(self) -> None:
        try:
            # Use LiveKit's utility to format messages for OpenAI-compatible API
            messages, _ = self._ctx.to_provider_format("openai")
            
            # Extract system instructions if audio is present
            system_instruction = ""
            has_audio = bool(self._llm.state.current_audio_b64)
            
            if has_audio:
                # Filter out system messages and collect content
                filtered_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system_instruction += msg.get("content", "") + "\n\n"
                    else:
                        filtered_messages.append(msg)
                messages = filtered_messages

            # Post-process to inject audio
    

            processed_messages = []
            
            # Identify the message object in context to update later
            # We look for the last user message with the placeholder
            # Note: We can't rely on reference equality with `messages` dicts because they are generated
            # We access the actual context object
            msg_to_update = None
            # ChatContext in this version uses _items to store messages
            chat_messages = getattr(self._ctx, "_items", [])
            if chat_messages:
                for msg in reversed(chat_messages):
                    if hasattr(msg, "role") and msg.role == "user":
                         msg_to_update = msg
                         break

            for i, msg in enumerate(messages):
                if msg.get("role") == "user" and msg.get("content") == "[AUDIO_INPUT]":
                     if i == len(messages) - 1:
                         if self._llm.state.current_audio_b64:
                             audio_b64 = self._llm.state.current_audio_b64
                             audio_len = len(audio_b64)
                             logger.info(f"Injecting audio data into LLM context. Length: {audio_len} chars")
                             
                             # Start background transcription
                             if msg_to_update:
                                 asyncio.create_task(self._transcribe_audio(audio_b64, msg_to_update))

                             content_list = []
                             
                             # Add system instruction as text part if available
                             if system_instruction:
                                 content_list.append({
                                     "type": "text",
                                     "text": system_instruction.strip()
                                 })
                             
                             # Add audio part
                             content_list.append({
                                    "type": "input_audio", 
                                    "input_audio": { 
                                        "data": audio_b64, 
                                        "format": "wav" 
                                    }
                                })
                                
                             msg["content"] = content_list
                             processed_messages.append(msg)
                         else:
                             logger.warning("Audio missing for current turn")
                             msg["content"] = "[Audio Missing]"
                             processed_messages.append(msg)
                     else:
                         # Not the last message, just set as text or keep placeholder
                         # Ideally this shouldn't happen if we updated it correctly in previous turns
                         msg["content"] = "[Previous Audio Input]"
                         processed_messages.append(msg)
                else:
                    processed_messages.append(msg)
            
            # Sanitize: Ensure no Tool message is immediately followed by a User message
            # Mistral/vLLM requires: Tool -> Assistant -> User
            sanitized_messages = []
            for i, msg in enumerate(processed_messages):
                sanitized_messages.append(msg)
                if msg.get("role") == "tool":
                    # Check if next message is User
                    if i + 1 < len(processed_messages):
                        next_msg = processed_messages[i+1]
                        if next_msg.get("role") == "user":
                            sanitized_messages.append({
                                "role": "assistant",
                                "content": "Tool execution complete." 
                            })
            processed_messages = sanitized_messages
            
            payload = {
                "model": self._llm.model,
                "messages": processed_messages,
                "tools": self._llm._tools if self._llm._tools else None
                
            }
            if not payload["tools"]: del payload["tools"]
            
            async with aiohttp.ClientSession() as session:
                url = f"{self._llm.base_url}/chat/completions"
                headers = {"Content-Type": "application/json"}
                async with session.post(url, headers=headers, json=payload) as resp:
                    resp_data = await resp.json()
                    
                    if "choices" in resp_data and len(resp_data["choices"]) > 0:
                        msg = resp_data["choices"][0]["message"]
                        content = msg.get("content")
                        tcs = msg.get("tool_calls")
                        
                        # Case 1: Native Tool Calls
                        if tcs:
                             for tc in tcs:
                                 chunk = llm.ChatChunk(
                                     id="voxtral_tc",
                                     delta=llm.ChoiceDelta(
                                         tool_calls=[llm.FunctionToolCall(
                                             call_id=tc["id"],
                                             name=tc["function"]["name"],
                                             arguments=tc["function"]["arguments"]
                                         )]
                                     )
                                 )
                                 await self._event_ch.send(chunk)

                        # Case 2: Content (Text or Encoded Tool Calls)
                        if content:
                             # Check for Voxtral's text-based tool calls: [TOOL_CALLS][{...}]
                             if content.strip().startswith("[TOOL_CALLS]"):
                                 try:
                                     json_str = content.strip().replace("[TOOL_CALLS]", "", 1)
                                     tool_payloads = json.loads(json_str)
                                     
                                     for i, tp in enumerate(tool_payloads):
                                         # Support flattened {"name":...} 
                                         fn_name = tp.get("name")
                                         fn_args = tp.get("arguments")
                                         
                                         # Handle potential nested format if any
                                         if not fn_name and "function" in tp:
                                             fn_name = tp["function"].get("name")
                                             fn_args = tp["function"].get("arguments")
                                         
                                         if isinstance(fn_args, dict):
                                             fn_args = json.dumps(fn_args)
                                             
                                         chunk = llm.ChatChunk(
                                             id=f"voxtral_tc_text_{i}",
                                             delta=llm.ChoiceDelta(
                                                 tool_calls=[llm.FunctionToolCall(
                                                     call_id=tp.get("id", uuid.uuid4().hex[:9]),
                                                     name=fn_name,
                                                     arguments=fn_args
                                                 )]
                                             )
                                         )
                                         await self._event_ch.send(chunk)
                                 except Exception as e:
                                     logger.error(f"Failed to parse text-based tool calls: {e}")
                                     chunk = llm.ChatChunk(
                                         id="voxtral_err",
                                         delta=llm.ChoiceDelta(content=content)
                                     )
                                     await self._event_ch.send(chunk)
                             else:
                                 chunk = llm.ChatChunk(
                                     id="voxtral_resp",
                                     delta=llm.ChoiceDelta(content=content)
                                 )
                                 await self._event_ch.send(chunk)
                    else:
                        logger.error(f"Voxtral LLM API returned unexpected: {resp_data}")

        except Exception as e:
            logger.error(f"Voxtral LLM request failed: {e}")
