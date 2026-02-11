#!/bin/bash
echo "Testing Indic Parler TTS Latency..."
echo "Input: 'Namaste, this is a test for low latency streaming.'"

curl -X POST "http://localhost:8890/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{"input": "Namaste, this is a test for low latency streaming.", "voice": "divya", "response_format": "pcm"}' \
     -o /dev/null \
     -s \
     -w "\n--- Timing Results ---\nDNS Lookup: %{time_namelookup}s\nConnect: %{time_connect}s\nTime to First Byte (TTFB): %{time_starttransfer}s\nTotal Download Time: %{time_total}s\n----------------------\n"
