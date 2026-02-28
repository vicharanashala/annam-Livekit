# Azure Speech TTS – Evaluation

## Voices Used
- Hindi: hi-IN
- Punjabi: pa-IN
- Bengali: bn-IN
- Gujarati: gu-IN
- Tamil: ta-IN

## Test Coverage
- Native scripts
- Roman scripts
- Mixed (Hinglish)
- Numerics (dates, phone numbers)

## Key Findings
- Native script synthesis works well across all tested languages.
- Roman script synthesis also works well across all tested languages.
- Mixed-language speech also works fine.

## Latency(in seconds)
- hi-IN hi native general latency: 1.599
- hi-IN hi native agri latency: 2.034
- hi-IN hi native numbers latency: 1.760
- hi-IN hi roman roman_numbers latency: 1.937
- pa-IN pa native general latency: 1.571
- pa-IN pa native numbers latency: 1.705
- pa-IN pa roman roman_numbers latency: 1.920
- bn-IN bn native general latency: 1.561
- bn-IN bn native numbers latency: 1.934
- bn-IN bn roman roman_numbers latency: 1.992
- gu-IN gu native general latency: 1.486
- gu-IN gu native numbers latency: 1.831
- gu-IN gu roman roman_numbers latency: 1.633
- ta-IN ta native general latency: 1.735
- ta-IN ta native numbers latency: 1.784
- ta-IN ta roman roman_numbers latency: 1.556
- hi-IN mix mixed hinglish latency: 1.284
