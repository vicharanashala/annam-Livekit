# Amazon Polly TTS – Evaluation

## Environment
- AWS Polly client region: ap-south-1
- Available voice(s) for India in our catalog: en-IN
- Attempted LanguageCode filters:
  - hi-IN returned 0 voices
  - bn-IN, gu-IN, pa-IN returned ValidationException (invalid LanguageCode)
  - ta-IN returned 0 voices
- Engine tested: standard

## Test Matrix
- Languages: Hindi, Punjabi, Bengali, Gujarati, Tamil
- Inputs: native script + roman script + mixed + numerics (date + phone)

## Observations (Audio)
### Native scripts
- bn/gu/pa/ta: FAIL
  - Output does not speak native words; primarily reads numerals (dates/phone) and skips native text.
- hi: PARTIAL
  - Reads native Hindi words but clarity is noticeably lower than Azure.
  - Native phone-number sentence is not spoken in a phone-number style (digit grouping issue).

### Roman scripts
- bn/gu/ta/pa: FAIL
  - Roman text is spoken with English accent/pronunciation (treated as English text).
- hi: OK for digits (phone/date) but accent is English.

## Verdict
Amazon Polly is not suitable for the multilingual Indian language requirement.
At best, it can be used for en-IN baseline or English-only flows.

## Latency(in seconds)
- hi native general -> amazon_polly_enIN_hi_native_general.wav | latency: 0.83 s
- hi native agri -> amazon_polly_enIN_hi_native_agri.wav | latency: 0.28 s
- hi native numbers -> amazon_polly_enIN_hi_native_numbers.wav | latency: 0.298 s
- hi roman roman_numbers -> amazon_polly_enIN_hi_roman_roman_numbers.wav | latency: 0.186 s
- pa native general -> amazon_polly_enIN_pa_native_general.wav | latency: 0.138 s
- pa native numbers -> amazon_polly_enIN_pa_native_numbers.wav | latency: 0.179 s
- pa roman roman_numbers -> amazon_polly_enIN_pa_roman_roman_numbers.wav | latency: 0.183 s
- bn native general -> amazon_polly_enIN_bn_native_general.wav | latency: 0.14 s
- bn native numbers -> amazon_polly_enIN_bn_native_numbers.wav | latency: 0.177 s
- bn roman roman_numbers -> amazon_polly_enIN_bn_roman_roman_numbers.wav | latency: 0.182 s
- gu native general -> amazon_polly_enIN_gu_native_general.wav | latency: 0.138 s
- gu native numbers -> amazon_polly_enIN_gu_native_numbers.wav | latency: 0.182 s
- gu roman roman_numbers -> amazon_polly_enIN_gu_roman_roman_numbers.wav | latency: 0.185 s
- ta native general -> amazon_polly_enIN_ta_native_general.wav | latency: 0.145 s
- ta native numbers -> amazon_polly_enIN_ta_native_numbers.wav | latency: 0.184 s
- ta roman roman_numbers -> amazon_polly_enIN_ta_roman_roman_numbers.wav | latency: 0.187 s
- mix mixed hinglish -> amazon_polly_enIN_mix_mixed_hinglish.wav | latency: 0.161 s
