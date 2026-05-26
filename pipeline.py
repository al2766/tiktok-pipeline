"""
@provenweird TikTok Video Pipeline
Input:  topic string
Output: single MP4 in ./output/ — no intermediates kept
Steps:  1. Claude (cached system prompt) → 12-sentence script + visual prompts
        2. ElevenLabs → voice MP3 + word timestamps
        3. FAL FLUX → 12 × 9:16 images
        4. FFmpeg → Ken Burns per image → concat → stretch → captions → mux audio
"""

import os, json, re, time, subprocess, base64, requests, asyncio, shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
FAL_KEY            = os.environ["FAL_KEY"]

def _ffmpeg() -> str:
    import shutil as _shutil
    exe = _shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

FFMPEG     = _ffmpeg()
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_W, VIDEO_H      = 1080, 1920
ELEVENLABS_VOICE_ID   = "nPczCjzI2devNBz1zQrb"  # Brian — deep, natural

# Cached once per worker — saves ~600 tokens on every script call
_SCRIPT_SYSTEM = """You write scripts for @provenweird — a faceless science facts TikTok engineered to keep 65%+ of viewers past the 3-second mark.

VOICE: Tom Scott meets dry stand-up. Confident, slightly sardonic, conversational authority. NOT excited. NOT a textbook. Sound like a brilliant person who finds most explanations inadequate and is fixing that — quickly.

CHOOSE THE BEST FORMULA for this topic:
- Formula A (PARADOX REVEAL): State the impossible-sounding fact upfront. Best for counterintuitive surface facts.
- Formula B (CURIOSITY GAP): Open the loop, don't answer it. Best when the MECHANISM is more interesting than the fact.
- Formula C (MYTH-BUST): "You've been told X. That's wrong." Best when a popular belief is false.
- Formula D (STAKES-FIRST): Open with the consequence affecting the viewer directly. Best for biology/psychology.
- Formula E (EXISTENTIAL PIVOT): Lead with the philosophical QUESTION the fact raises — not the fact itself. Best when the fact implies something about identity, consciousness, or time.

STRUCTURE (exactly 12 sentences, minimum 60 seconds of speech):
1.  HOOK — The scroll-stopper. NEVER "Did you know". Max 15 words.
2-3. DESTABILISE — Confirm the hook is real. Add a layer that makes it even stranger.
4.  SPECIFIC NUMBER — One exact measurement or statistic. Anchors credibility.
5-9. MECHANISM — Explain WHY, fast. Max 12 words per sentence. Plain language, no passive voice.
10. HUMAN COMPARISON — Scale to something absurd and relatable. The shareable moment.
11. TWIST — Most unexpected implication. The thing they didn't see coming.
12. KICKER — Dry callback to the hook. The line people screenshot and comment.

NON-NEGOTIABLE RULES:
- NEVER open with "Did you know", "Today we're", "It has been", or any passive opener
- NEVER end on a question to the viewer
- Exactly one specific number/measurement — not zero, not two
- Total script: 90-130 words (60-70 seconds at natural pace)
- Second-person where possible ("your", "you", "imagine")
- Halal content only. No music references.

Return ONLY valid JSON, no markdown, no backticks:
{"hook":"first sentence only","sentences":["exactly 12 strings"],"visual_prompts":["exactly 12 cinematic image prompts — photorealistic or hyper-detailed illustration, 9:16 vertical, vivid and dramatic, directly illustrates the sentence, no text in image"],"tiktok_caption":"punchy 1-line caption with the wildest fact + 5 hashtags"}"""


# ─── Step 1: Script ───────────────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    print(f"[1/4] Generating script: {topic}")
    for attempt in range(3):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1800,
                "system": [{"type": "text", "text": _SCRIPT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": f"Topic: {topic}"}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        # Strip markdown fences if Claude adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            data = json.loads(raw.strip())
            data["script"] = " ".join(data["sentences"])
            print(f"   Hook: {data['hook']}")
            return data
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError("Script generation failed after 3 attempts")


# ─── Step 2: TTS + word timestamps ───────────────────────────────────────────

def generate_tts(script: str, slug: str) -> tuple[Path, list[dict]]:
    print("[2/4] Generating voice...")
    audio_path = OUTPUT_DIR / f"{slug}_audio.mp3"

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/with-timestamps",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.85, "style": 0.2, "use_speaker_boost": True},
        },
        timeout=60,
    )

    if resp.status_code == 401:
        print("   ElevenLabs blocked — falling back to edge-tts + Whisper")
        _edge_tts(script, audio_path)
        boundaries = _whisper_timestamps(audio_path)
    else:
        resp.raise_for_status()
        data = resp.json()
        audio_path.write_bytes(base64.b64decode(data["audio_base64"]))
        boundaries = _parse_elevenlabs_alignment(data.get("alignment", {}))

    print(f"   {len(boundaries)} word boundaries")
    return audio_path, boundaries


def _parse_elevenlabs_alignment(alignment: dict) -> list[dict]:
    chars  = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends   = alignment.get("character_end_times_seconds", [])
    boundaries, word, w_start, w_end = [], "", None, None
    for ch, s, e in zip(chars, starts, ends):
        if ch in (" ", "\n"):
            if word:
                boundaries.append({"word": word, "start": round(w_start, 3), "end": round(w_end, 3)})
                word, w_start, w_end = "", None, None
        else:
            if w_start is None:
                w_start = s
            w_end = e
            word += ch
    if word:
        boundaries.append({"word": word, "start": round(w_start, 3), "end": round(w_end, 3)})
    return boundaries


def _edge_tts(script: str, audio_path: Path):
    import edge_tts
    async def _run():
        comm = edge_tts.Communicate(script, voice="en-US-ChristopherNeural")
        await comm.save(str(audio_path))
    asyncio.run(_run())


def _whisper_timestamps(audio_path: Path) -> list[dict]:
    from faster_whisper import WhisperModel
    model_dir = Path(__file__).parent / "whisper_models"
    model_dir.mkdir(exist_ok=True)
    model = WhisperModel("small", device="cpu", compute_type="int8", download_root=str(model_dir))
    segments_raw, _ = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)
    boundaries = []
    for seg in segments_raw:
        if seg.words:
            for w in seg.words:
                boundaries.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
    del model
    return boundaries


# ─── Step 3: Images via FAL FLUX ─────────────────────────────────────────────

def generate_images(visual_prompts: list[str], slug: str) -> list[Path]:
    print(f"[3/4] Generating {len(visual_prompts)} images...")
    img_dir = OUTPUT_DIR / f"{slug}_img"
    img_dir.mkdir(exist_ok=True)
    paths = []
    for i, prompt in enumerate(visual_prompts):
        resp = requests.post(
            "https://fal.run/fal-ai/flux/schnell",
            headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
            json={
                "prompt": prompt + ", 9:16 vertical aspect ratio, ultra high quality, cinematic",
                "image_size": "portrait_4_3",
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": False,
            },
            timeout=90,
        )
        resp.raise_for_status()
        url = resp.json()["images"][0]["url"]
        img_path = img_dir / f"{i:02d}.jpg"
        img_path.write_bytes(requests.get(url, timeout=60).content)
        print(f"   {i+1}/{len(visual_prompts)} ✓")
        paths.append(img_path)
    return paths


# ─── Step 4: Assemble ─────────────────────────────────────────────────────────

_ZOOMPAN = [
    "min(zoom+0.0008,1.12)",
    "if(lte(zoom,1.0),1.10,max(1.0,zoom-0.0008))",
]


def _audio_duration(audio_path: Path) -> float:
    r = subprocess.run([FFMPEG, "-i", str(audio_path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        raise RuntimeError("Could not read audio duration")
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _ass_time(t: float) -> str:
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = t % 60
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def assemble_video(image_paths: list[Path], audio_path: Path, boundaries: list[dict], slug: str) -> Path:
    print("[4/4] Assembling video...")
    tmp = OUTPUT_DIR / f"{slug}_tmp"
    tmp.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{slug}.mp4"

    # Ken Burns per image → 5-second clips at 25fps
    clip_paths = []
    for i, img in enumerate(image_paths):
        z    = _ZOOMPAN[i % 2]
        clip = tmp / f"c{i:02d}.mp4"
        vf   = (
            f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
            f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"zoompan=z='{z}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
            f":d=125:s={VIDEO_W}x{VIDEO_H}:fps=25"
        )
        r = subprocess.run(
            [FFMPEG, "-y", "-loop", "1", "-i", str(img),
             "-vf", vf, "-frames:v", "125",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
             "-tune", "zerolatency", "-threads", "1", "-pix_fmt", "yuv420p",
             str(clip)],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg clip {i+1} failed:\n{r.stderr.decode(errors='replace')[-400:]}")
        clip_paths.append(clip)
        print(f"   Clip {i+1}/{len(image_paths)} animated")

    # Concat clips
    concat_list = tmp / "clips.txt"
    with open(concat_list, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c.resolve()}'\n")

    silent = tmp / "silent.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(silent)],
        check=True, capture_output=True,
    )

    # Stretch or trim to match audio
    audio_dur = _audio_duration(audio_path)
    total_dur = len(clip_paths) * 5.0
    stretched = tmp / "stretched.mp4"
    if total_dur < audio_dur:
        pts = 1.0 / (total_dur / audio_dur)
        subprocess.run(
            [FFMPEG, "-y", "-i", str(silent),
             "-vf", f"setpts={pts:.4f}*PTS",
             "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-threads", "1", "-an",
             str(stretched)],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            [FFMPEG, "-y", "-i", str(silent), "-t", str(audio_dur), "-c", "copy", str(stretched)],
            check=True, capture_output=True,
        )

    # Build ASS karaoke captions
    font = "Impact" if any(
        os.path.exists(p) for p in ["/Library/Fonts/Impact.ttf", "/System/Library/Fonts/Impact.ttf"]
    ) else "Helvetica"
    ass = tmp / "captions.ass"
    with open(ass, "w", encoding="utf-8") as f:
        f.write(f"[Script Info]\nPlayResX: {VIDEO_W}\nPlayResY: {VIDEO_H}\nScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{font},72,&H00FFFFFF,&H00FFFFFF,"
                f"&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,4,0,2,30,30,160,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        i = 0
        while i < len(boundaries):
            grp   = boundaries[i:i+3]
            parts = [
                f"{{\\k{max(1, int((w['end'] - w['start']) * 100))}}}{w['word'].upper()}"
                for w in grp
            ]
            f.write(
                f"Dialogue: 0,{_ass_time(grp[0]['start'])},{_ass_time(grp[-1]['end'])},"
                f"Default,,0,0,0,,{' '.join(parts)}\n"
            )
            i += 3

    # Final mux: stretched video + audio + burnt captions
    subprocess.run(
        [FFMPEG, "-y",
         "-i", str(stretched), "-i", str(audio_path),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
         "-tune", "zerolatency", "-threads", "2",
         "-vf", (
             f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
             f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2,"
             f"ass={ass}"
         ),
         "-c:a", "aac", "-b:a", "192k", "-shortest",
         str(out_path)],
        check=True, capture_output=True,
    )

    # Clean up everything except the final MP4
    shutil.rmtree(tmp, ignore_errors=True)
    img_dir = image_paths[0].parent if image_paths else None
    for p in image_paths:
        p.unlink(missing_ok=True)
    if img_dir and img_dir.exists():
        shutil.rmtree(img_dir, ignore_errors=True)
    audio_path.unlink(missing_ok=True)

    print(f"✅ {out_path.name}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_video(topic: str) -> tuple[Path, str]:
    slug        = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"
    script      = generate_script(topic)
    audio, wds  = generate_tts(script["script"], slug)
    images      = generate_images(script["visual_prompts"], slug)
    video       = assemble_video(images, audio, wds, slug)
    caption     = script.get("tiktok_caption", "")
    print(f"\n📋 Caption: {caption}")
    return video, caption


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "honey never expires — found in 3000-year-old Egyptian tombs"
    generate_video(topic)
