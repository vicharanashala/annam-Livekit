# Azure Speech TTS – Evaluation

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
- Native script synthesis works well across all tested languages.
- Punjabi (Gurmukhi) works when explicit voice is set.
- Mixed-language speech shows accent bias depending on voice.

## Latency(in seconds)
- hi-IN hi native general latency: 1.784
- hi-IN hi native agri latency: 1.802
- hi-IN hi native numbers latency: 1.814
- hi-IN hi roman roman_numbers latency: 1.937
- pa-IN pa native general latency: 1.956
- pa-IN pa native numbers latency: 2.451
- pa-IN pa roman roman_numbers latency: 2.137
- bn-IN bn native general latency: 1.708
- bn-IN bn native numbers latency: 2.087
- bn-IN bn roman roman_numbers latency: 2.046
- gu-IN gu native general latency: 1.917
- gu-IN gu native numbers latency: 2.527
- gu-IN gu roman roman_numbers latency: 2.646
- ta-IN ta native general latency: 1.69
- ta-IN ta native numbers latency: 2.029
- ta-IN ta roman roman_numbers latency: 1.941
- hi-IN en mixed hinglish latency: 1.643
