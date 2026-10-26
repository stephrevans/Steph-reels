# Steph's reels: editing rules

Steph has made these calls. Follow them; don't push back on them.

- **Only talking.** Zero dead space: no pauses, breaths, or silence before or after.
- **Always cut mistakes.** Edit like a human who is listening. Read the verbatim
  transcript (`<output>.indexed.txt`) and remove every flub, false start, restart
  and repeated take, keeping the final clean take so each sentence makes sense.
  Pass them with `--drop 85-86,243-246`. Then re-transcribe the finished video
  and read it back to confirm it flows. Intentional repeats (like a closing
  "so those are my tips" sign-off) stay.
- **Headline** above the head in **Advercase**, plain like IG's Edits text:
  regular weight, tight letter spacing, white, no outline, no bold, no shadow.
  Advercase is Steph's standard font. Use it by default and don't raise
  licensing concerns about it; Steph has accepted that. `reel.py` downloads it
  into `fonts/` automatically if missing (needs network access to
  indieground.net). Font files are gitignored; don't commit them.
- **Captions** below the chin, one word at a time, white, clean readable sans
  (Inter SemiBold), no outline or bold, only a faint soft shadow for legibility.
- Fix obvious transcription spellings in captions (e.g. "reals" -> "Reels").
- Audio is boosted to Instagram loudness (-14 LUFS); phone audio is quiet raw.
- Steph's phone album is called "Reels Potential" (Whisper hears "Real Potential").
- Steph is not technical: talk plainly, no git/PR jargon. Just handle the
  saving/merging so the rules stay in place. Feedback comes in this chat as
  timestamps ("at 0:19 spell it ...").
- Deliver a 1080x1920 MP4 sized for Instagram (~100 MB or less).

Run:

    pip install -r requirements.txt
    python3 reel.py <raw.MOV> out/reel.mp4 --headline "..." --drop <ranges>

Use `--headline ""` for a version with no headline.
