#!/usr/bin/env python3
"""Turn a raw talking-head clip into a reel.

1. Cuts out every silence.
2. Puts a headline above your head (Advercase font) for the whole video.
3. Adds one-word-at-a-time white captions just below your chin.

Usage:
  python3 reel.py raw/IMG_7803.MOV out/reel.mp4 \
      --headline "content tricks I've done for 6 years"

Needs: ffmpeg, `pip install faster-whisper opencv-python-headless`,
and the Advercase font file in fonts/ (any .otf/.ttf with "advercase" in the name).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def probe(path):
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", path]).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return {
        "w": w, "h": h,
        "duration": float(info["format"]["duration"]),
        "hdr": v.get("color_transfer") in ("arib-std-b67", "smpte2084"),
    }


# ---------- 1. cutting to speech only ----------

def cut_silences(src, dst, keep, hdr):
    # select/aselect stream frame by frame, so memory stays low even on 4K.
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
    v = f"select='{expr}',setpts=N/FRAME_RATE/TB,scale=1080:-2"
    if hdr:  # iPhone HDR -> normal SDR so colors don't look washed out on IG
        v += (",zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
              "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
    v += ",format=yuv420p"
    graph = f"[0:v]{v}[v];[0:a]aselect='{expr}',asetpts=N/SR/TB[a]"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(graph)
    run(["ffmpeg", "-y", "-hide_banner", "-i", src, "-filter_complex_script", f.name,
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
         "-r", "30", "-c:a", "aac", "-b:a", "192k", dst])
    os.unlink(f.name)


# ---------- 2. find head + chin ----------

def face_lines(path, w, h, duration):
    """Return (top_of_head_y, chin_y) in pixels, from the median face over the clip."""
    import cv2
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    tops, chins = [], []
    for k in range(15):
        cap.set(cv2.CAP_PROP_POS_MSEC, duration * 1000 * (k + 0.5) / 15)
        ok, frame = cap.read()
        if not ok:
            continue
        fh = frame.shape[0]
        scale = h / fh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 6, minSize=(fh // 10, fh // 10))
        if len(faces):
            x, y, fw, fhh = max(faces, key=lambda r: r[2] * r[3])
            tops.append((y - 0.30 * fhh) * scale)   # haar box starts at forehead
            chins.append((y + 1.10 * fhh) * scale)  # box ends near chin
    cap.release()
    if not tops:
        return 0.22 * h, 0.55 * h
    tops.sort(); chins.sort()
    return tops[len(tops) // 2], chins[len(chins) // 2]


# ---------- 3. captions ----------

SR = 16000


def load_audio(path):
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def speech_ranges(audio, words, duration, floor_db=None, blocked=()):
    """Keep only spots that are both loud enough to be voice AND inside a
    transcribed word. Everything else (pauses, breaths, rustling) is cut."""
    import numpy as np
    hop = int(SR * 0.01)  # 10 ms frames
    n = len(audio) // hop
    rms = np.sqrt(np.mean(audio[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    if floor_db is None:  # halfway between background noise and speech
        floor_db = (np.percentile(db, 10) + np.percentile(db, 90)) / 2
    loud = db > floor_db

    def grow(mask, k):  # extend each True run by k frames both ways
        idx = np.flatnonzero(mask)
        out = np.zeros_like(mask)
        for i in idx:
            out[max(0, i - k):i + k + 1] = True
        return out

    def close(mask, k):  # fill False gaps shorter than k frames
        m = mask.copy()
        i = 0
        while i < len(m):
            if not m[i]:
                j = i
                while j < len(m) and not m[j]:
                    j += 1
                if 0 < i and j < len(m) and j - i < k:
                    m[i:j] = True
                i = j
            else:
                i += 1
        return m

    loud = grow(close(loud, 8), 5)          # bridge tiny dips, 50 ms padding
    in_word = np.zeros(n, bool)
    for a, b, _ in words:
        in_word[max(0, int((a - 0.04) * 100)):int((b + 0.08) * 100)] = True
    for a, b in blocked:                    # flubs/restarts cut by the editor
        in_word[max(0, int(a * 100)):int(b * 100)] = False
    keep = close(loud & in_word, 6)         # don't chop between words < 60 ms
    ranges, i = [], 0
    while i < n:
        if keep[i]:
            j = i
            while j < n and keep[j]:
                j += 1
            if j - i >= 8:                  # ignore blips under 80 ms
                ranges.append((i / 100, min(j / 100, duration)))
            i = j
        else:
            i += 1
    return ranges, floor_db


def remap(words, ranges):
    """Move word times from the raw video onto the cut video's timeline."""
    out, offs = [], []
    acc = 0.0
    for a, b in ranges:
        offs.append(acc)
        acc += b - a
    for a, b, t in words:
        for (ra, rb), o in zip(ranges, offs):
            if rb > a and ra < b:  # word overlaps this piece
                out.append((o + max(a, ra) - ra, o + min(b, rb) - ra, t))
                break
    return out


def transcribe(audio, model_size):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # The prompt nudges Whisper to write down ums, stutters and restarts
    # instead of tidying them away, so flubs show up in the transcript.
    segs, _ = model.transcribe(
        audio, word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
        initial_prompt="Umm, so, uh, I- I was like, um, the- the thing is, uh, you know.")
    words = []
    for s in segs:
        for wd in s.words:
            t = wd.word.strip()
            if t:
                words.append((wd.start, wd.end, t))
    return words


def ts(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def esc(t):
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def headline_text(text, font, missing):
    """Escape the headline; draw characters the font lacks (e.g. the
    apostrophe in the Advercase demo) in a similar serif instead."""
    out = ""
    for ch in text:
        if ch in missing:
            if ch == "'":
                ch = "\u2019"
            out += f"{{\\fnFreeSerif}}{ch}{{\\fn{font}}}"
        else:
            out += esc(ch)
    return out


def write_ass(path, w, h, duration, words, headline, head_font, cap_font,
              head_top, chin, head_bold=False, missing=""):
    head_size = int(w * 0.085)
    cap_size = int(w * 0.072)
    margin = int(w * 0.07)
    # Headline: bottom edge sits a little above the top of the head.
    head_y = max(int(head_top - h * 0.02), head_size * 2 + int(h * 0.04))
    # Caption: top edge a little below the chin.
    cap_y = int(chin + h * 0.025)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Clean white text, no outline, no shadow, tight tracking (IG Edits look).
        f"Style: Head,{head_font},{head_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"{-1 if head_bold else 0},0,0,0,100,100,{-head_size * 0.045:.1f},0,1,0,0,2,{margin},{margin},0,1",
        # Captions: same clean look plus a faint blurred shadow so they stay
        # readable over bright things (paper, window). Not an outline.
        f"Style: Cap,{cap_font},{cap_size},&H00FFFFFF,&H00FFFFFF,&H9A000000,&H9A000000,"
        f"0,0,0,0,100,100,0,0,1,{max(3, cap_size // 16)},0,8,{margin},{margin},0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if headline:  # pass --headline "" to leave it off (e.g. to add it in Edits)
        lines.append(f"Dialogue: 0,{ts(0)},{ts(duration)},Head,,0,0,0,,"
                     f"{{\\an2\\pos({w // 2},{head_y})}}{headline_text(headline, head_font, missing)}")
    for i, (a, b, t) in enumerate(words):
        nxt = words[i + 1][0] if i + 1 < len(words) else b + 0.3
        end = min(max(b, a + 0.12), nxt) if nxt > a else b
        pop = "\\blur6"
        lines.append(f"Dialogue: 1,{ts(a)},{ts(end)},Cap,,0,0,0,,"
                     f"{{\\an8\\pos({w // 2},{cap_y}){pop}}}{esc(t)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


ADVERCASE_URL = "https://indieground.net/product/indieground-advercasefont-demo.zip"


def find_advercase(folder):
    files = [f for f in glob.glob(os.path.join(folder, "*"))
             if "advercase" in f.lower() and f.lower().endswith((".otf", ".ttf"))]
    # Prefer the Regular weight (the plain IG Edits look).
    return sorted(files, key=lambda f: ("regular" not in f.lower(), f))


def fetch_advercase(folder):
    """Download Advercase into fonts/ (it isn't kept in git)."""
    import io
    import urllib.request
    import zipfile
    try:
        req = urllib.request.Request(ADVERCASE_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if base.lower().endswith(".otf") and not name.startswith("__MACOSX"):
                    with open(os.path.join(folder, base), "wb") as f:
                        f.write(z.read(name))
        print("Downloaded Advercase into", folder)
    except Exception as e:
        print("Couldn't download Advercase:", e)


def font_family(font_file):
    out = run(["fc-scan", "--format", "%{family[0]}", font_file]).stdout.strip()
    return out or os.path.splitext(os.path.basename(font_file))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--headline", default="content tricks I've done for 6 years")
    ap.add_argument("--fonts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))
    ap.add_argument("--caption-font", default="Inter SemiBold")
    ap.add_argument("--headline-bold", action="store_true")
    ap.add_argument("--fallback-headline-font", default="Inter Display Black")
    ap.add_argument("--floor-db", type=float, default=None,
                    help="voice loudness threshold; default is picked from the audio")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--drop", default="",
                    help="flubbed words to cut, as transcript index ranges like 85-86,243-246 "
                         "(see <output>.indexed.txt)")
    args = ap.parse_args()

    adv = find_advercase(args.fonts)
    if not adv:
        fetch_advercase(args.fonts)
        adv = find_advercase(args.fonts)
    missing = ""
    if adv:
        head_font = font_family(adv[0])
        from fontTools.ttLib import TTFont
        cmap = TTFont(adv[0]).getBestCmap()
        missing = "".join(sorted({c for c in args.headline if c != " " and ord(c) not in cmap}))
        if missing:
            print(f"{head_font} has no {missing!r}; drawing those in a serif")
    else:
        head_font = args.fallback_headline_font
        print(f"No Advercase file in {args.fonts}/ - using {head_font} for the headline")

    info = probe(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    work = tempfile.mkdtemp()
    base = os.path.splitext(args.output)[0]

    audio = load_audio(args.input)
    wjson = base + ".words.json"
    if os.path.exists(wjson):
        words = [tuple(x) for x in json.load(open(wjson))]
    else:
        words = transcribe(audio, args.model)
        json.dump(words, open(wjson, "w"))
    words = [(a, b, re.sub(r"[.,!?;:]+$", "", t)) for a, b, t in words]
    print(f"Transcribed {len(words)} words")
    with open(base + ".indexed.txt", "w") as f:  # for picking --drop ranges
        f.write(" ".join(f"{i}:{t}" for i, (_, _, t) in enumerate(words)) + "\n")

    # Drop flubbed takes: remove those words and block their audio, from the
    # end of the last kept word before them to the start of the next kept one.
    drop = set()
    for part in filter(None, args.drop.split(",")):
        lo, _, hi = part.partition("-")
        drop.update(range(int(lo), int(hi or lo) + 1))
    blocked = []
    for i in sorted(drop):
        if i - 1 in drop:
            continue
        j = i
        while j + 1 in drop:
            j += 1
        a = words[i - 1][1] + 0.02 if i > 0 else 0.0
        b = words[j + 1][0] - 0.03 if j + 1 < len(words) else info["duration"]
        blocked.append((a, b))
        print("Cutting flub:", " ".join(t for _, _, t in words[i:j + 1]))
    words = [w for i, w in enumerate(words) if i not in drop]

    keep, floor = speech_ranges(audio, words, info["duration"], args.floor_db, blocked)
    kept = sum(b - a for a, b in keep)
    print(f"Voice threshold {floor:.1f} dB; keeping {len(keep)} pieces: "
          f"{info['duration']:.1f}s -> {kept:.1f}s")
    cut = os.path.join(work, "cut.mp4")
    cut_silences(args.input, cut, keep, info["hdr"])
    words = remap(words, keep)
    with open(base + ".txt", "w") as f:
        f.write(" ".join(t for _, _, t in words) + "\n")

    cinfo = probe(cut)
    w, h = cinfo["w"], cinfo["h"]
    head_top, chin = face_lines(cut, w, h, cinfo["duration"])
    print(f"Head top ~{head_top:.0f}px, chin ~{chin:.0f}px (of {h})")

    ass = os.path.join(work, "subs.ass")
    write_ass(ass, w, h, cinfo["duration"], words, args.headline, head_font,
              args.caption_font, head_top, chin, args.headline_bold, missing)
    run(["ffmpeg", "-y", "-hide_banner", "-i", cut,
         "-vf", f"ass={ass}:fontsdir={args.fonts}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "8M",
         "-bufsize", "12M", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", args.output])
    print("Done:", args.output)


if __name__ == "__main__":
    main()
