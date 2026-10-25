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


# ---------- 1. silence cutting ----------

def find_keep_ranges(path, duration, noise_db, min_silence, pad):
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        text=True, capture_output=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    if len(ends) < len(starts):
        ends.append(duration)
    keep, cur = [], 0.0
    for s, e in zip(starts, ends):
        a, b = cur, min(s + pad, duration)
        if b - a > 0.05:
            keep.append((a, b))
        cur = max(e - pad, 0.0)
    if duration - cur > 0.05:
        keep.append((cur, duration))
    return keep


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

def transcribe(path, model_size):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(path, word_timestamps=True, vad_filter=False)
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


def write_ass(path, w, h, duration, words, headline, head_font, cap_font,
              head_top, chin):
    head_size = int(w * 0.085)
    cap_size = int(w * 0.075)
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
        f"Style: Head,{head_font},{head_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
        f"0,0,0,0,100,100,0,0,1,{max(2, head_size // 22)},{max(2, head_size // 18)},2,{margin},{margin},0,1",
        f"Style: Cap,{cap_font},{cap_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(3, cap_size // 14)},{max(2, cap_size // 20)},8,{margin},{margin},0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 0,{ts(0)},{ts(duration)},Head,,0,0,0,,{{\\an2\\pos({w // 2},{head_y})}}{esc(headline)}",
    ]
    for i, (a, b, t) in enumerate(words):
        nxt = words[i + 1][0] if i + 1 < len(words) else b + 0.3
        end = min(max(b, a + 0.12), nxt) if nxt > a else b
        pop = "\\fscx85\\fscy85\\t(0,80,\\fscx100\\fscy100)"
        lines.append(f"Dialogue: 1,{ts(a)},{ts(end)},Cap,,0,0,0,,"
                     f"{{\\an8\\pos({w // 2},{cap_y}){pop}}}{esc(t)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def font_family(font_file):
    out = run(["fc-scan", "--format", "%{family[0]}", font_file]).stdout.strip()
    return out or os.path.splitext(os.path.basename(font_file))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--headline", default="content tricks I've done for 6 years")
    ap.add_argument("--fonts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))
    ap.add_argument("--caption-font", default="Inter ExtraBold")
    ap.add_argument("--fallback-headline-font", default="Inter Display Black")
    ap.add_argument("--silence-db", type=float, default=-35)
    ap.add_argument("--min-silence", type=float, default=0.30)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()

    adv = [f for f in glob.glob(os.path.join(args.fonts, "*"))
           if "advercase" in f.lower() and f.lower().endswith((".otf", ".ttf"))]
    if adv:
        head_font = font_family(adv[0])
    else:
        head_font = args.fallback_headline_font
        print(f"No Advercase file in {args.fonts}/ - using {head_font} for the headline")

    info = probe(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    work = tempfile.mkdtemp()
    cut = os.path.splitext(args.output)[0] + ".cut.mp4"
    if os.path.exists(cut):
        print("Reusing", cut)
    else:
        keep = find_keep_ranges(args.input, info["duration"], args.silence_db,
                                args.min_silence, args.pad)
        kept = sum(b - a for a, b in keep)
        print(f"Keeping {len(keep)} pieces: {info['duration']:.1f}s -> {kept:.1f}s")
        cut_silences(args.input, cut, keep, info["hdr"])

    cinfo = probe(cut)
    w, h = cinfo["w"], cinfo["h"]
    head_top, chin = face_lines(cut, w, h, cinfo["duration"])
    print(f"Head top ~{head_top:.0f}px, chin ~{chin:.0f}px (of {h})")

    words = transcribe(cut, args.model)
    words = [(a, b, re.sub(r"[.,!?;:]+$", "", t)) for a, b, t in words]
    print(f"Transcribed {len(words)} words")
    with open(os.path.splitext(args.output)[0] + ".txt", "w") as f:
        f.write(" ".join(t for _, _, t in words) + "\n")

    ass = os.path.join(work, "subs.ass")
    write_ass(ass, w, h, cinfo["duration"], words, args.headline, head_font,
              args.caption_font, head_top, chin)
    run(["ffmpeg", "-y", "-hide_banner", "-i", cut,
         "-vf", f"ass={ass}:fontsdir={args.fonts}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-movflags", "+faststart", args.output])
    print("Done:", args.output)


if __name__ == "__main__":
    main()
