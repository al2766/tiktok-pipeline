"""
Reel Content Intelligence Pipeline
===================================
Input:  any reel/video URL (Instagram, TikTok, YouTube, etc.)
Output: JSON with full word-level transcript + matching video frames

Each entry in the output map:
  {
    "word":       "something",
    "start":      4.32,          # seconds from video start
    "end":        4.71,
    "segment":    "Full sentence this word belongs to",
    "frame_path": "/path/to/frame_4.32.jpg"  # null if frame extraction skipped
  }

Usage:
  from reel_analyzer import analyze_reel
  result = analyze_reel("https://www.instagram.com/reels/DYhwnqcxPKu/")
  # result["words"]    → list of word entries above
  # result["segments"] → list of full segments with start/end
  # result["full_text"] → plain transcript string
  # result["frames_dir"] → directory containing extracted frames
"""

import os, json, subprocess, re, shutil, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OpenMP conflict on macOS
from pathlib import Path
from faster_whisper import WhisperModel

WORK_DIR   = Path(__file__).parent / "analysis_cache"
MODELS_DIR = Path(__file__).parent / "whisper_models"
FFMPEG_EXE = shutil.which("ffmpeg") or os.environ.get("FFMPEG_BINARY", "ffmpeg")

WORK_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# "base" model — ~150 MB, good accuracy, fast on CPU (~30s for a 60s reel)
# Upgrade to "small" (~500 MB) for better accuracy on noisy audio
_MODEL: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        print("[analyzer] Loading Whisper small model (first run downloads ~500 MB)…")
        _MODEL = WhisperModel("small", device="cpu", compute_type="int8",
                              download_root=str(MODELS_DIR))
        print("[analyzer] Model loaded.")
    return _MODEL


def _slug(url: str) -> str:
    """Turn a URL into a safe filesystem slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", url)
    return clean[-60:] + f"_{int(time.time())}"


def _download_media(url: str, out_dir: Path) -> tuple[Path, Path]:
    """
    Download video + audio from URL using yt-dlp + Chrome cookies.
    Returns (video_path, audio_path).
    yt-dlp uses the logged-in Chrome session so private/login-walled content works.
    """
    print(f"[analyzer] Downloading media from: {url}")

    video_path = out_dir / "video.mp4"
    audio_path = out_dir / "audio.mp3"

    # Try with Chrome cookies first (for Instagram login), fall back without
    cookie_args = [
        "--cookies-from-browser", "chrome",
        "--quiet", "--no-warnings",
    ]

    ydl_cmd_video = [
        "yt-dlp",
        *cookie_args,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
        url,
    ]

    ydl_cmd_audio = [
        "yt-dlp",
        *cookie_args,
        "-x", "--audio-format", "mp3",
        "-o", str(audio_path),
        url,
    ]

    # Download video
    result = subprocess.run(ydl_cmd_video, capture_output=True, text=True)
    if result.returncode != 0:
        # Retry without cookies (public content)
        result = subprocess.run(
            [c for c in ydl_cmd_video if c not in ("--cookies-from-browser", "chrome")],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp video download failed: {result.stderr[:500]}")

    # Download audio separately for cleaner Whisper input
    result = subprocess.run(ydl_cmd_audio, capture_output=True, text=True)
    if result.returncode != 0:
        result = subprocess.run(
            [c for c in ydl_cmd_audio if c not in ("--cookies-from-browser", "chrome")],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        # Fall back: extract audio from the video we already have
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", str(video_path),
             "-vn", "-acodec", "libmp3lame", "-q:a", "4", str(audio_path)],
            check=True, capture_output=True
        )

    print(f"[analyzer] Downloaded: video={video_path.exists()} audio={audio_path.exists()}")
    return video_path, audio_path


def _transcribe(audio_path: Path) -> tuple[list[dict], list[dict], str]:
    """
    Run faster-whisper with word-level timestamps.
    Returns (words, segments, full_text).
    """
    print("[analyzer] Transcribing audio…")
    model = _get_model()

    segments_raw, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,          # skip silence — cleaner output
        language=None,            # auto-detect
    )

    words    = []
    segments = []
    full_text_parts = []

    for seg in segments_raw:
        seg_text = seg.text.strip()
        full_text_parts.append(seg_text)
        segments.append({
            "text":  seg_text,
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
        })
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":    w.word.strip(),
                    "start":   round(w.start, 3),
                    "end":     round(w.end, 3),
                    "segment": seg_text,
                })

    print(f"[analyzer] Transcribed {len(words)} words, {len(segments)} segments.")
    return words, segments, " ".join(full_text_parts)


def _extract_frames(video_path: Path, timestamps: list[float], frames_dir: Path) -> dict[float, str]:
    """
    Extract one frame per unique timestamp using ffmpeg.
    Returns dict of {timestamp: frame_path}.
    """
    frames_dir.mkdir(exist_ok=True)
    frame_map = {}

    # Deduplicate timestamps to avoid re-extracting the same frame
    seen = set()
    unique_ts = []
    for t in timestamps:
        bucket = round(t, 1)  # group within 0.1s to avoid redundant frames
        if bucket not in seen:
            seen.add(bucket)
            unique_ts.append((t, bucket))

    print(f"[analyzer] Extracting {len(unique_ts)} frames from video…")

    for ts, bucket in unique_ts:
        out_path = frames_dir / f"frame_{bucket:.1f}.jpg"
        if out_path.exists():
            frame_map[bucket] = str(out_path)
            continue
        result = subprocess.run(
            [FFMPEG_EXE, "-y", "-ss", str(ts), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "3", str(out_path)],
            capture_output=True
        )
        if result.returncode == 0:
            frame_map[bucket] = str(out_path)

    print(f"[analyzer] Extracted {len(frame_map)} frames.")
    return frame_map


def analyze_reel(url: str, extract_frames: bool = True) -> dict:
    """
    Full pipeline: download → transcribe → match words to frames.

    Returns:
      {
        "url":        str,
        "full_text":  str,
        "segments":   [{text, start, end}],
        "words":      [{word, start, end, segment, frame_path}],
        "frames_dir": str | None,
        "cache_dir":  str,
      }
    """
    slug     = _slug(url)
    out_dir  = WORK_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached result
    cache_file = out_dir / "result.json"
    if cache_file.exists():
        print(f"[analyzer] Returning cached result from {out_dir}")
        with open(cache_file) as f:
            return json.load(f)

    # Step 1: Download
    video_path, audio_path = _download_media(url, out_dir)

    # Step 2: Transcribe
    words, segments, full_text = _transcribe(audio_path)

    # Step 3: Extract frames matching each word's start time
    frames_dir = None
    if extract_frames and video_path.exists() and words:
        frames_dir = out_dir / "frames"
        timestamps = [w["start"] for w in words]
        frame_map  = _extract_frames(video_path, timestamps, frames_dir)

        # Attach frame_path to each word
        for w in words:
            bucket = round(w["start"], 1)
            w["frame_path"] = frame_map.get(bucket)

        frames_dir = str(frames_dir)

    result = {
        "url":        url,
        "full_text":  full_text,
        "segments":   segments,
        "words":      words,
        "frames_dir": frames_dir,
        "cache_dir":  str(out_dir),
    }

    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[analyzer] Done. Cached at {out_dir}")
    return result


def print_summary(result: dict):
    """Pretty-print a result for terminal review."""
    print(f"\n{'='*60}")
    print(f"URL: {result['url']}")
    print(f"\nFULL TRANSCRIPT:\n{result['full_text']}")
    print(f"\nSEGMENTS ({len(result['segments'])}):")
    for s in result["segments"]:
        print(f"  [{s['start']:.1f}s → {s['end']:.1f}s] {s['text']}")
    print(f"\nWORDS ({len(result['words'])}) — first 10:")
    for w in result["words"][:10]:
        frame = Path(w["frame_path"]).name if w.get("frame_path") else "no frame"
        print(f"  {w['start']:.2f}s  \"{w['word']}\"  → {frame}")
    print(f"\nFrames dir: {result['frames_dir']}")
    print("="*60)


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.instagram.com/reels/DYhwnqcxPKu/"
    result = analyze_reel(url)
    print_summary(result)
