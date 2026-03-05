# Google Cloud Chirp TTS – Evaluation

## Voices Used
- Hindi: hi-IN-Chirp-A
- Punjabi: pa-IN-Chirp-A
- Bengali: bn-IN-Chirp-A
- Gujarati: gu-IN-Chirp-A
- Tamil: ta-IN-Chirp-A

## Test Coverage
- Native scripts
- Roman scripts
- Mixed (Hinglish)
- Numerics (dates, phone numbers)

## Key Findings
- performed reliably across all tested Indian languages, with especially strong results in Bengali and Gujarati.
- The main limitation observed was numeric handling(mobile numbers were typically read as full numbers instead of digit-by-digit) mainly in Hindi, Punjabi and Tamil.

## Latency(in seconds)
- hi-IN hi native general latency: 0.700
- hi-IN hi native agri latency: 0.831
- hi-IN hi native numbers latency: 1.298
- hi-IN hi roman roman_numbers latency: 1.592
- pa-IN pa native general latency: 0.673
- pa-IN pa native numbers latency: 1.058
- pa-IN pa roman roman_numbers latency: 0.859
- bn-IN bn native general latency: 0.710
- bn-IN bn native numbers latency: 1.729
- bn-IN bn roman roman_numbers latency: 1.776
- gu-IN gu native general latency: 0.887
- gu-IN gu native numbers latency: 1.723
- gu-IN gu roman roman_numbers latency: 1.763
- ta-IN ta native general latency: 0.948
- ta-IN ta native numbers latency: 1.625
- ta-IN ta roman roman_numbers latency: 1.690
- hi-IN en mixed hinglish latency: 0.694
