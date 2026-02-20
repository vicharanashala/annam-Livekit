1. IndicConformer

17/02/2026 Conclusion – Evaluation of IndicConformer (CTC vs RNNT)

The IndicConformer multilingual STT model was tested for Malayalam speech recognition using both CTC (Connectionist Temporal Classification) and RNNT (Recurrent Neural Network Transducer) decoding strategies.

🔹 Overall Performance

Both decoders successfully transcribed Malayalam speech with high accuracy. The generated outputs were grammatically meaningful and very close to the reference sentences.

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

The model does not generate punctuation such as full stops. This is expected because:

ASR models transcribe acoustic speech.

Full stops and commas are not explicitly spoken.

Punctuation restoration requires a separate post-processing model.

🎯 Final Summary

Both CTC and RNNT performed well for Malayalam speech recognition.

RNNT demonstrated better contextual understanding but showed slight variability across speakers.

CTC was faster but slightly less context-aware.

Minor word merging and splitting variations were observed in both methods.

Overall transcription quality is high and suitable for practical applications with optional post-processing (punctuation restoration).
# ..............

18/02/2026

2. Openai Whisper Large V3: Transcribe Audio (Whisper latest)
   
Tested the latest OpenAI Whisper model using whisper.load_model("large-v3") on Malayalam audio files.

However, the results were not accurate. The audio was not transcribed correctly in Malayalam. For one or two short audio samples, it partially transcribed the content in Malayalam, but the transcription was still incomplete and inaccurate.

In several other cases, the model transcribed the Malayalam audio into different languages. Sometimes, instead of performing transcription, the model automatically translated the audio into another language.
# ................
19/02/2026

3. facebook/mms-1b-all
    
Completed testing on facebook/mms-1b-all using Malayalam audio samples. A comparison was conducted between Indic Conformer and facebook/mms-1b-all.

For short audio clips, mms-1b-all performs well with minor errors. However, for longer audio samples, it produces more errors.

# ......................
20/02/2026

4. "openai/whisper-large-v3-turbo"

testing with  model_id = "openai/whisper-large-v3-turbo"

but the audio is not transcribed properly
model keeps guessing language , also repetition bug and hallucination




