# Eleven Labs TTS – Evaluation

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
- Dates were not pronounced properly, they were read as raw numeric number.

## Latency(in seconds)
- hi-IN hi native general latency: 1.478
- hi-IN hi native agri latency: 2.079
- hi-IN hi native numbers latency: 2.560
- hi-IN hi roman roman_numbers latency: 2.533
- pa-IN pa native general latency: 2.592
- pa-IN pa native numbers latency: 3.259
- pa-IN pa roman roman_numbers latency: 2.621
- bn-IN bn native general latency: 1.785
- bn-IN bn native numbers latency: 3.341
- bn-IN bn roman roman_numbers latency: 2.660
- gu-IN gu native general latency: 2.584
- gu-IN gu native numbers latency: 2.987
- gu-IN gu roman roman_numbers latency: 2.555
- ta-IN ta native general latency: 2.518
- ta-IN ta native numbers latency: 2.980
- ta-IN ta roman roman_numbers latency: 2.477
- hi-IN mix mixed hinglish latency: 1.465
