1. indic_parler_tts
tested with dfeerent malyalam prompts
for small promts its working alsmost
for lengthy its not working well, unnatural sound, voice breaking

2. indicf5_tts
tested indicf5_tts using different prompt Malayalam, Hindi, English
The output voice is not good: breaking the sentences, not cleared
retried after changing sample rate , lengthy audio refernce text:  a little bit difference. Not upto the expectation level
also tried the same audio sound .wav input and its corresponding reference script from the indicf5 github repo
but same effect

13/02/2026

The new IndicF5 script, generate_indicf5_tts_wav, shared by Shubankar was tested with different Malayalam voice samples to identify a native-sounding speaker. The sound2_female reference audio and its corresponding text were selected for further processing. The model performs well for short prompts; however, for longer prompts, the voice quality degrades and noticeable noise appears at the beginning of the generated audio.

14/02/2026
A lengthy text prompt (around four paragraphs) was tested using different input audio samples ranging from very short (single sentence) to longer clips (approximately five sentences).
First, a very short reference audio was used with the same long text prompt. It did not work properly — the output contained unnatural sounds and noise.
Next, a slightly longer reference audio was tried. The output improved, but the quality was still not satisfactory.
Then, a longer voice sample was tested, which worked much better.
However, when the reference audio length was increased further (very long audio), the output quality again degraded, especially for very long sentences, producing unclear and distorted speech.
From these observations, it appears that a moderate-length reference audio (neither too short nor too long) works best for both short and long text prompts.
Therefore, there may be a limitation related to reference audio length in IndicF5, which requires further investigation.
