#!/bin/bash

# Endpoint URL
URL="http://localhost:8890/v1/audio/speech"

# Output file
OUTPUT_FILE="test_output.pcm"

echo "Sending request to $URL..."
echo "Saving audio to $OUTPUT_FILE"

# Time the request
time curl -X POST "$URL" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "ai4bharat/indic-parler-tts",
           "input": "This is a test of the streaming functionality with torchaudio resampling.",
           "voice": "divya",
           "response_format": "pcm",
           "speed": 1.0
         }' \
     --output "$OUTPUT_FILE"

echo "Done. Audio saved to $OUTPUT_FILE"
echo "To play (if you have ffplay installed):"
echo "ffplay -f s16le -ar 24000 -ac 1 $OUTPUT_FILE"
