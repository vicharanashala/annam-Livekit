Explored Vibe Coding (prompt-based AI coding) and Claude Code (structured AI-assisted development for large codebases).
Completed Qwen TTS 0.6B fine-tuning (10 epochs) successfully.
Loss reduced from 12.65 → 2.72
Saved 10 complete checkpoints (standalone models)
Ran inference on final checkpoint:
Very slow (25–30 mins per sentence)
Generated ~11 min audio for 14 words → model not stopping properly
Applied max_new_tokens to control length:
Reduced audio to ~39 sec
But audio quality remained unclear
Identified key issues:
Improper end-of-sequence learning
Poor audio clarity
Root cause (hypothesis):
Added projection layer (2048 → 1024) is randomly initialized and distorting outputs
Likely architecture mismatch between 0.6B model and 1.7B fine-tuning script
Next steps:
More inference testing