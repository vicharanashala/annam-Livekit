#!/bin/bash

# Test curl command for wav_server.py
# This script sends a TTS request and saves the output as a WAV file

# Server URL
SERVER_URL="http://localhost:8000"

# Text to synthesize
TEXT="नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."

# Output file
OUTPUT_PCM="output.pcm"
OUTPUT_WAV="output.wav"

echo "Sending TTS request..."

# Send request and save raw PCM data
curl -X POST "${SERVER_URL}/tts" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"${TEXT}\"}" \
  --output "${OUTPUT_PCM}"

echo "Converting PCM to WAV..."

# Convert PCM to WAV using ffmpeg
# -f s16le: signed 16-bit little-endian PCM
# -ar 24000: sample rate 24kHz
# -ac 1: mono (1 channel)
ffmpeg -f s16le -ar 24000 -ac 1 -i "${OUTPUT_PCM}" -y "${OUTPUT_WAV}"

echo "Audio saved to ${OUTPUT_WAV}"
echo "Cleaning up temporary PCM file..."
rm -f "${OUTPUT_PCM}"

echo "Done! You can play the audio with: ffplay ${OUTPUT_WAV}"

echo ""
echo "---------------------------------------------------"
echo "Testing new /tts_wav endpoint (Direct WAV download)"
echo "---------------------------------------------------"

OUTPUT_DIRECT_WAV="direct_output.wav"

curl -X POST "${SERVER_URL}/tts_wav" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"${TEXT}\"}" \
  --output "${OUTPUT_DIRECT_WAV}"

echo "Direct WAV saved to ${OUTPUT_DIRECT_WAV}"
echo "You can play it with: ffplay ${OUTPUT_DIRECT_WAV}"
