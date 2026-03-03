# Google Cloud WaveNet TTS – Evaluation

## Voices Used
- Hindi: hi-IN-Wavenet-A
- Punjabi: pa-IN-Wavenet-A
- Bengali: bn-IN-Wavenet-A
- Gujarati: gu-IN-Wavenet-A
- Tamil: ta-IN-Wavenet-A

## Test Coverage
- Native scripts
- Roman scripts
- Mixed (Hinglish)
- Numerics (dates, phone numbers)

## Key Findings
- Results showed much better latency than Google Cloud Standard Model.
- Performed reliably across all tested Indian languages, with especially strong results in Bengali and Gujarati.
- The main limitation observed was numeric handling(mobile numbers were typically read as full numbers instead of digit-by-digit).
- Roman Hindi dates sometimes carried an English accent.

## Latency(in seconds)
- hi-IN hi native general latency: 0.132
- hi-IN hi native agri latency: 0.131
- hi-IN hi native numbers latency: 0.246
- hi-IN hi roman roman_numbers latency: 0.389
- pa-IN pa native general latency: 0.133
- pa-IN pa native numbers latency: 0.246
- pa-IN pa roman roman_numbers latency: 0.307
- bn-IN bn native general latency: 0.115
- bn-IN bn native numbers latency: 0.270
- bn-IN bn roman roman_numbers latency: 0.287
- gu-IN gu native general latency: 0.227
- gu-IN gu native numbers latency: 0.245
- gu-IN gu roman roman_numbers latency: 0.246
- ta-IN ta native general latency: 0.138
- ta-IN ta native numbers latency: 0.269
- ta-IN ta roman roman_numbers latency: 0.375
- hi-IN en mixed hinglish latency: 0.173
