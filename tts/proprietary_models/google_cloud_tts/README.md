# Google Cloud TTS – Evaluation

## Voices Used
- Hindi: hi-IN-Standard-A
- Punjabi: pa-IN-Standard-A
- Bengali: bn-IN-Standard-A
- Gujarati: gu-IN-Standard-A
- Tamil: ta-IN-Standard-A

## Test Coverage
- Native scripts
- Roman scripts
- Mixed (Hinglish)
- Numerics (dates, phone numbers)

## Key Findings
- performed reliably across all tested Indian languages, with especially strong results in Bengali and Gujarati.
- The main limitation observed was numeric handling(mobile numbers were typically read as full numbers instead of digit-by-digit).
- Roman Hindi dates sometimes carried an English accent.

## Latency(in seconds)
- hi-IN hi native general latency: 0.204
- hi-IN hi native agri latency: 0.297
- hi-IN hi native numbers latency: 0.331
- hi-IN hi roman roman_numbers latency: 0.374
- pa-IN pa native general latency: 0.199
- pa-IN pa native numbers latency: 0.234
- pa-IN pa roman roman_numbers latency: 0.177
- bn-IN bn native general latency: 0.237
- bn-IN bn native numbers latency: 0.350
- bn-IN bn roman roman_numbers latency: 0.344
- gu-IN gu native general latency: 0.283
- gu-IN gu native numbers latency: 0.468
- gu-IN gu roman roman_numbers latency: 0.234
- ta-IN ta native general latency: 0.315
- ta-IN ta native numbers latency: 0.574
- ta-IN ta roman roman_numbers latency: 0.471
- hi-IN en mixed hinglish latency: 0.623
