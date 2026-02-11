#!/bin/bash
# Test Indic TTS API

echo "Waiting for server to start on port 8027..."
timeout 30 sh -c 'until nc -z localhost 8027; do sleep 1; done'

echo "Server is up! Sending health check..."
curl -v http://localhost:8027/health

echo -e "\n\nTesting Synthesis (Hindi)..."
curl -X POST "http://localhost:8027/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "सोचने की आज़ादी का मतलब यह नहीं कि हर सोच सही हो,
और सवाल उठाने का साहस ही समाज को आगे बढ़ाता है।", "lang": "hi", "speaker": "female"}' \
     --output test_api_output1.mp3

echo -e "\n\nSynthesis complete. Checking output file..."
ls -lh test_api_output1.mp3
file test_api_output1.mp3
