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

        return VoxtralLLMStream(self, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, tools=tools_list, tool_choice=tool_choice)

class VoxtralLLMStream(llm.LLMStream):
    def __init__(self, llm_inst: VoxtralLLM, chat_ctx: llm.ChatContext, fnc_ctx: Optional[llm.ToolContext], tools: List[Union[llm.FunctionTool, llm.RawFunctionTool]], tool_choice: Union[llm.ToolChoice, None] = None):
        super().__init__(llm=llm_inst, chat_ctx=chat_ctx, tools=tools, conn_options=APIConnectOptions())
        self._llm = llm_inst
        self._ctx = chat_ctx
        self._tool_choice = tool_choice
        
    async def _run(self) -> None:
        try:
            # Use LiveKit's utility to format messages for OpenAI-compatible API
            messages, _ = self._ctx.to_provider_format("openai")
            
            # Check if audio is present
            has_audio = bool(self._llm.state.current_audio_b64)

            # Post-process to inject audio
            processed_messages = []
            for i, msg in enumerate(messages):
                if msg.get("role") == "user" and msg.get("content") == "[AUDIO_INPUT]":
                     if i == len(messages) - 1:
                         if self._llm.state.current_audio_b64:
                             audio_len = len(self._llm.state.current_audio_b64)
                             logger.info(f"Injecting audio data into LLM context. Length: {audio_len} chars")
                             
                             content_list = []
                             
                             # Add audio part
                             content_list.append({
                                    "type": "input_audio", 
                                    "input_audio": { 
                                        "data": self._llm.state.current_audio_b64, 
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
                         msg["content"] = "[Previous Audio Input]"
                         processed_messages.append(msg)
                else:
                    processed_messages.append(msg)
            
            # Sanitize: Ensure no Tool message is immediately followed by a User message
            # Mistral/vLLM requires: Tool -> Assistant -> User
            sanitized_messages = []
            for i, msg in enumerate(processed_messages):
                # Ensure tool arguments are valid JSON strings
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args = func.get("arguments")
                        if isinstance(args, dict):
                            func["arguments"] = json.dumps(args)
                        elif args is None or (isinstance(args, str) and not args.strip()):
                            func["arguments"] = "{}"
                
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
            logger.info(f"Final messages: {json.dumps(processed_messages)}")
            
            payload = {
                "model": self._llm.model,
                "messages": processed_messages,
                "tools": self._llm._tools if self._llm._tools else None,
                "stream": True
            }
            if self._tool_choice is not None and type(self._tool_choice).__name__ != "NotGiven":
                payload["tool_choice"] = self._tool_choice
            if not payload["tools"]: del payload["tools"]
            
            async with aiohttp.ClientSession() as session:
                url = f"{self._llm.base_url}/chat/completions"
                headers = {"Content-Type": "application/json"}
                async with session.post(url, headers=headers, json=payload) as resp:
                    # Buffer for text-based tool calls (Voxtral specific)
                    buffer = ""
                    possible_tool_call = True
                    tool_call_cache = {}
                    
                    async for line in resp.content:
                        line_text = line.decode("utf-8").strip()
                        if not line_text.startswith("data: "):
                            continue
                        
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            # Flush any pending native tool calls
                            for idx, tool_data in tool_call_cache.items():
                                chunk = llm.ChatChunk(
                                    id=f"voxtral_tc_{idx}",
                                    delta=llm.ChoiceDelta(
                                        tool_calls=[llm.FunctionToolCall(
                                            call_id=tool_data["id"],
                                            name=tool_data["name"],
                                            arguments=tool_data["arguments"]
                                        )]
                                    )
                                )
                                await self._event_ch.send(chunk)
                            break
                            
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            
                            # 1. Native Tool Calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                possible_tool_call = False # Native detected
                                for tc in delta["tool_calls"]:
                                    idx = tc["index"]
                                    if idx not in tool_call_cache:
                                        tool_call_cache[idx] = {"id": "", "name": "", "arguments": ""}
                                    
                                    if tc.get("id"):
                                        tool_call_cache[idx]["id"] = tc["id"]
                                    if tc.get("function"):
                                        if tc["function"].get("name"):
                                            tool_call_cache[idx]["name"] = tc["function"]["name"]
                                        if tc["function"].get("arguments"):
                                            tool_call_cache[idx]["arguments"] += tc["function"]["arguments"]

                            # Check for finish_reason to flush native tool calls
                            if data["choices"][0].get("finish_reason"):
                                for idx, tool_data in tool_call_cache.items():
                                    chunk = llm.ChatChunk(
                                        id=f"voxtral_tc_{idx}",
                                        delta=llm.ChoiceDelta(
                                            tool_calls=[llm.FunctionToolCall(
                                                call_id=tool_data["id"],
                                                name=tool_data["name"],
                                                arguments=tool_data["arguments"]
                                            )]
                                        )
                                    )
                                    await self._event_ch.send(chunk)
                                tool_call_cache = {} # Clear after flushing

                            # 2. Content (Text or Text-based Tool Calls)
                            content = delta.get("content")
                            if content:
                                if possible_tool_call:
                                    buffer += content
                                    prefix = "[TOOL_CALLS]"
                                    # If buffer definitely strictly diverges from prefix, flush and disable buffering
                                    if len(buffer) <= len(prefix):
                                        if not prefix.startswith(buffer):
                                            possible_tool_call = False
                                            chunk = llm.ChatChunk(
                                                 id="voxtral_resp",
                                                 delta=llm.ChoiceDelta(content=buffer)
                                            )
                                            await self._event_ch.send(chunk)
                                            buffer = ""
                                    else:
                                        if not buffer.startswith(prefix):
                                            possible_tool_call = False
                                            chunk = llm.ChatChunk(
                                                 id="voxtral_resp",
                                                 delta=llm.ChoiceDelta(content=buffer)
                                            )
                                            await self._event_ch.send(chunk)
                                            buffer = ""
                                else:
                                    # Standard streaming
                                    chunk = llm.ChatChunk(
                                         id="voxtral_resp",
                                         delta=llm.ChoiceDelta(content=content)
                                     )
                                    await self._event_ch.send(chunk)
                                    
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing stream line: {e}")

                    # 3. Post-stream processing
                    if possible_tool_call and buffer:
                         # Try parsing as tool calls
                         if buffer.strip().startswith("[TOOL_CALLS]"):
                             try:
                                 json_str = buffer.strip().replace("[TOOL_CALLS]", "", 1)
                                 tool_payloads = json.loads(json_str)
                                 
                                 for i, tp in enumerate(tool_payloads):
                                     fn_name = tp.get("name")
                                     fn_args = tp.get("arguments")
                                     
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
                                     delta=llm.ChoiceDelta(content=buffer)
                                 )
                                 await self._event_ch.send(chunk)
                         else:
                             # It was just buffered text
                             chunk = llm.ChatChunk(
                                 id="voxtral_resp",
                                 delta=llm.ChoiceDelta(content=buffer)
                             )
                             await self._event_ch.send(chunk)

        except Exception as e:
            logger.error(f"Voxtral LLM request failed: {e}")