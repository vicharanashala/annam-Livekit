# Amazon Polly TTS – Evaluation

## Voices Used
- Hindi: hi-IN-AaravNeural
- Punjabi: pa-IN-OjasNeural
- Bengali: bn-IN-TanishaaNeural
- Gujarati: gu-IN-DhwaniNeural
- Tamil: ta-IN-PallaviNeural

## Test Coverage
- Native scripts
- Roman scripts
- Mixed (Hinglish)
- Numerics (dates, phone numbers)

## Key Findings
- Native script synthesis does not work well across all tested languages. Output does not speak native words; primarily reads numerals (dates/phone) and skips native text. For hindi, the model reads native words but the clarity is noticeably lower.
- roman script synthesis works ok across all tested languages. Roman text is spoken with english accent.
- Mixed-language speech shows accent bias towards english.

## Latency(in seconds)
- hi-IN hi native general latency: 0.83 s
- hi-IN hi native agri latency: 0.28 s
- hi-IN hi native numbers latency: 0.298 s
- hi-IN hi roman roman_numbers latency: 0.186 s
- pa-IN pa native general latency: 0.138 s
- pa-IN pa native numbers latency: 0.179 s
- pa-IN pa roman roman_numbers latency: 0.183 s
- bn-IN bn native general latency: 0.14 s
- bn-IN bn native numbers latency: 0.177 s
- bn-IN bn roman roman_numbers latency: 0.182 s
- gu-IN gu native general latency: 0.138 s
- gu-IN gu native numbers latency: 0.182 s
- gu-IN gu roman roman_numbers latency: 0.185 s
- ta-IN ta native general latency: 0.145 s
- ta-IN ta native numbers latency: 0.184 s
- ta-IN ta roman roman_numbers latency: 0.187 s
- hi-IN en mixed hinglish latency: 0.161 s
