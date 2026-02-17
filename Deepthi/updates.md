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

17/02/2026
Conclusion – Evaluation of IndicConformer (CTC vs RNNT)

The IndicConformer multilingual STT model was tested for Malayalam speech recognition using both CTC (Connectionist Temporal Classification) and RNNT (Recurrent Neural Network Transducer) decoding strategies.

🔹 Overall Performance

Both decoders successfully transcribed Malayalam speech with high accuracy.
The generated outputs were grammatically meaningful and very close to the reference sentences.

🔹 Observations on CTC

CTC produced fast and reasonably accurate results.

Minor variations were observed in word formation.

In some cases, joined Malayalam words were transcribed correctly.

Since CTC predicts tokens independently per time frame, it may slightly truncate suffixes or merge/split words based purely on acoustic signals.

CTC is suitable for:

Faster inference

Lightweight deployment

Clean and short audio samples

🔹 Observations on RNNT

RNNT generally produced linguistically stronger outputs.

Because RNNT uses previous predictions as context, it handles morphological structures better.

However, in one instance, RNNT misread a word slightly.

Also, when the same sentence was spoken by different speakers, RNNT showed small variations in output.

In one case, a joined Malayalam word was transcribed as two separate words.

This indicates:

RNNT is sensitive to speaker variation (pitch, accent, pronunciation).

Though context-aware, it may still produce minor inconsistencies across speakers.

RNNT is suitable for:

Production systems

Longer sentences

Morphologically rich languages like Malayalam

Context-sensitive transcription

🔹 Speaker Variation Observation

When the same audio content was spoken by different speakers:

Minor transcription differences were observed.

Some joined words were separated.

Slight morphological variations appeared.

This suggests:

The model captures acoustic differences between speakers.

Pronunciation style affects token prediction.

Speaker normalization could further improve consistency.

🔹 Punctuation Observation

The model does not generate punctuation such as full stops.
This is expected because:

ASR models transcribe acoustic speech.

Full stops and commas are not explicitly spoken.

Punctuation restoration requires a separate post-processing model.

🎯 Final Summary

Both CTC and RNNT performed well for Malayalam speech recognition.

RNNT demonstrated better contextual understanding but showed slight variability across speakers.

CTC was faster but slightly less context-aware.

Minor word merging and splitting variations were observed in both methods.

Overall transcription quality is high and suitable for practical applications with optional post-processing (punctuation restoration).
