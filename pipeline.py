"""
@provenweird TikTok Video Pipeline — v2
Input:  topic string
Output: MP4 file in ./output/

Pipeline:
  1. Claude  → 10-sentence script + per-sentence visual prompts
  2. ElevenLabs → natural voice audio + word-level timestamps
  3. FAL AI (FLUX) → 1 HD 9:16 image per sentence
  4. Kling AI → animate each image into a 5s video clip
  5. FFmpeg  → concat clips + mux audio + burn animated captions
"""

import os, json, re, time, subprocess, base64, gc, requests, textwrap, asyncio
import jwt
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import fal_client

load_dotenv()

ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
KLING_ACCESS_KEY   = os.environ["KLING_ACCESS_KEY"]
KLING_SECRET_KEY   = os.environ["KLING_SECRET_KEY"]
FAL_KEY            = os.environ["FAL_KEY"]
os.environ["FAL_KEY"] = FAL_KEY

import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_W, VIDEO_H = 1080, 1920
ELEVENLABS_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # Brian — deep, natural, male


# ─── Fonts ────────────────────────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── Step 1: Script ───────────────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    print(f"[1/5] Generating script: {topic}")
    system = """You write scripts for @provenweird — a faceless science facts TikTok. Your scripts are engineered to keep 65%+ of viewers past the 3-second mark. The algorithm punishes weak hooks and rewards completion.

VOICE: Tom Scott meets dry stand-up. Confident, slightly sardonic, conversational authority. NOT excited. NOT a textbook. Sound like a brilliant person who finds most explanations inadequate and is going to fix that — quickly.

CHOOSE THE BEST FORMULA for this topic and apply it:
- Formula A (PARADOX REVEAL): State the impossible-sounding fact upfront. Best for counterintuitive surface facts.
- Formula B (CURIOSITY GAP): Open the loop, don't answer it. Best when the MECHANISM is more interesting than the fact.
- Formula C (MYTH-BUST): "You've been told X. That's wrong." Best when a popular belief is false.
- Formula D (STAKES-FIRST): Open with the consequence affecting the viewer directly. Best for biology/psychology.
- Formula E (EXISTENTIAL PIVOT): Lead with the philosophical QUESTION the fact raises — not the fact itself. The science is the evidence, not the hook. Best when the fact implies something about identity, consciousness, existence, or time. Example: instead of "your body replaces its cells every 7 years" → "If your body replaces most of its cells over time, are you still the same person?" The question opens a loop inside the viewer's own head. They save it because the question is the point.

STRUCTURE (exactly 12 sentences — minimum 60 seconds for TikTok Creator Rewards):
1.  HOOK — The scroll-stopper. State the paradox, gap, myth, or stakes. NEVER "Did you know". Max 15 words.
2-3. DESTABILISE — Confirm the hook is real. Add a layer that makes it even stranger.
4.  SPECIFIC NUMBER — One exact measurement or statistic. This anchors credibility. (e.g. "exactly 37 degrees", "900 million tonnes", "3 times per second")
5-9. MECHANISM — Explain WHY, fast. Max 12 words per sentence. Plain language, no passive voice. Each sentence lands alone.
10. HUMAN COMPARISON — Scale it to something absurd and relatable. This is the shareable moment.
11. TWIST — The most unexpected implication. The thing they didn't see coming.
12. KICKER — Dry callback to the hook. Reframes everything. This is the line people screenshot and comment.

NON-NEGOTIABLE RULES:
- NEVER open with "Did you know", "Today we're", "It has been", or any passive opener
- NEVER end on a question to the viewer ("What do you think?")
- Exactly one specific number/measurement — not zero, not two
- Total script: 90-130 words (60-70 seconds of speech at natural pace)
- Second-person where possible ("your", "you", "imagine")
- Halal content only. No music references.

Return ONLY valid JSON (no markdown, no backticks):
{
  "hook": "first sentence only — the scroll-stopper",
  "sentences": ["exactly 12 strings"],
  "visual_prompts": ["exactly 12 cinematic image prompts — one per sentence, photorealistic or hyper-detailed illustration, 9:16 vertical, vivid and dramatic, directly illustrates the sentence, no text in image"],
  "tiktok_caption": "punchy 1-line caption with the wildest fact + 5 hashtags — no exclamation spam"
}"""

    for attempt in range(3):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 2000,
                "system": system,
                "messages": [{"role": "user", "content": f"Topic: {topic}"}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            data = json.loads(raw.strip())
            # Build full script string from sentences
            data["script"] = " ".join(data["sentences"])
            print(f"   Hook: {data['hook']}")
            return data
        except json.JSONDecodeError:
            time.sleep(2)
    raise RuntimeError("Script generation failed")


# ─── Step 2: TTS with word timestamps ────────────────────────────────────────

def _whisper_word_boundaries(audio_path: Path) -> list[dict]:
    """Run faster-whisper on audio to extract word-level timestamps."""
    from faster_whisper import WhisperModel
    model_dir = Path(__file__).parent / "whisper_models"
    model_dir.mkdir(exist_ok=True)
    print("   Running Whisper for word timestamps...")
    model = WhisperModel("small", device="cpu", compute_type="int8",
                         download_root=str(model_dir))
    segments_raw, _ = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)
    boundaries = []
    for seg in segments_raw:
        if seg.words:
            for w in seg.words:
                boundaries.append({
                    "word":  w.word.strip(),
                    "start": round(w.start, 3),
                    "end":   round(w.end, 3),
                })
    del model
    return boundaries


def _elevenlabs_tts(script: str, audio_path: Path) -> list[dict] | None:
    """Try ElevenLabs TTS. Returns word boundaries on success, None on 401."""
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/with-timestamps",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.85,
                "style": 0.2,
                "use_speaker_boost": True,
            },
        },
        timeout=60,
    )
    if resp.status_code == 401:
        return None
    resp.raise_for_status()
    data = resp.json()
    audio_path.write_bytes(base64.b64decode(data["audio_base64"]))

    boundaries, alignment = [], data.get("alignment", {})
    chars  = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends   = alignment.get("character_end_times_seconds", [])

    word, w_start, w_end = "", None, None
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


def _edge_tts_generate(script: str, audio_path: Path):
    import edge_tts
    async def _run():
        comm = edge_tts.Communicate(script, voice="en-US-ChristopherNeural")
        await comm.save(str(audio_path))
    asyncio.run(_run())


def generate_tts(script: str, slug: str) -> tuple[Path, list[dict]]:
    audio_path = OUTPUT_DIR / f"{slug}_audio.mp3"

    print("[2/5] Generating voice with ElevenLabs...")
    boundaries = _elevenlabs_tts(script, audio_path)

    if boundaries is None:
        # ElevenLabs blocked (free tier / VPN detection) — fall back to edge-tts
        print("   ElevenLabs unavailable — falling back to edge-tts + Whisper timestamps")
        _edge_tts_generate(script, audio_path)
        boundaries = _whisper_word_boundaries(audio_path)

    print(f"   {len(boundaries)} word boundaries, audio saved.")
    return audio_path, boundaries


# ─── Step 3: FAL AI images (one per sentence) ─────────────────────────────────

def generate_images(visual_prompts: list[str], slug: str) -> list[Path]:
    """Generate images via FAL FLUX and save locally. Returns list of local paths."""
    print(f"[3/5] Generating {len(visual_prompts)} images via FAL FLUX...")
    img_dir = OUTPUT_DIR / f"{slug}_images"
    img_dir.mkdir(exist_ok=True)
    image_paths = []

    for i, prompt in enumerate(visual_prompts):
        print(f"   Image {i+1}/{len(visual_prompts)}...")
        result = fal_client.run(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt + ", 9:16 vertical aspect ratio, ultra high quality",
                "image_size": "portrait_4_3",
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": False,
            },
        )
        url = result["images"][0]["url"]
        img_path = img_dir / f"img_{i:02d}.jpg"
        img_data = requests.get(url, timeout=60).content
        img_path.write_bytes(img_data)
        image_paths.append(img_path)
        print(f"   ✓ saved {img_path.name}")

    return image_paths


# ─── Step 4: Kling image-to-video ─────────────────────────────────────────────

def _kling_jwt() -> str:
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256",
                      headers={"alg": "HS256", "typ": "JWT"})


def _kling_submit(image_path: Path, motion_prompt: str) -> str:
    """Submit image-to-video task. Sends image as base64 to avoid CDN restrictions."""
    token = _kling_jwt()
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    resp = requests.post(
        "https://api.klingai.com/v1/videos/image2video",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model_name": "kling-v1",
            "image": img_b64,
            "prompt": motion_prompt,
            "duration": "5",
            "mode": "std",
            "aspect_ratio": "9:16",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Kling submit failed: {data}")
    task_id = data["data"]["task_id"]
    print(f"   Submitted task {task_id}")
    return task_id


def _kling_poll(task_id: str, timeout: int = 300) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(10)
        token = _kling_jwt()
        resp = requests.get(
            f"https://api.klingai.com/v1/videos/image2video/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling poll error: {data}")
        task = data["data"]
        status = task.get("task_status", "")
        if status == "succeed":
            return task["task_result"]["videos"][0]["url"]
        if status == "failed":
            raise RuntimeError(f"Kling task failed: {task.get('task_status_msg')}")
        print(f"   Kling {task_id}: {status}...")
    raise RuntimeError(f"Kling task {task_id} timed out")


MOTION_PROMPTS = [
    "slow cinematic zoom in, gentle camera drift",
    "slow pull back, soft ambient motion",
    "subtle camera pan left, dreamy atmosphere",
    "slow zoom out revealing the scene",
    "gentle handheld camera drift, cinematic",
    "slow dolly forward, ethereal glow",
    "camera slowly rotates, atmospheric haze",
    "subtle zoom in, particles floating",
    "gentle camera float upward, cinematic light",
    "slow pan right, soft depth of field",
]


KLING_CONCURRENCY = 5  # trial pack limit


def animate_images(image_paths: list[Path], slug: str) -> list[Path]:
    print(f"[4/5] Animating {len(image_paths)} images with Kling...")
    clip_dir = OUTPUT_DIR / f"{slug}_clips"
    clip_dir.mkdir(exist_ok=True)

    video_paths = []
    for batch_start in range(0, len(image_paths), KLING_CONCURRENCY):
        batch = image_paths[batch_start: batch_start + KLING_CONCURRENCY]
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        task_ids = []
        for i, img_path in zip(batch_indices, batch):
            motion = MOTION_PROMPTS[i % len(MOTION_PROMPTS)]
            print(f"   Submitting clip {i+1}/{len(image_paths)}...")
            task_id = _kling_submit(img_path, motion)
            task_ids.append((i, task_id))
            time.sleep(1)

        for i, task_id in task_ids:
            print(f"   Waiting for clip {i+1}/{len(image_paths)} (task {task_id})...")
            video_url = _kling_poll(task_id)
            clip_path = clip_dir / f"clip_{i:02d}.mp4"
            r = requests.get(video_url, timeout=120)
            r.raise_for_status()
            clip_path.write_bytes(r.content)
            print(f"   ✓ Clip {i+1} saved ({len(r.content)//1024}KB)")
            video_paths.append(clip_path)

    return video_paths


# ─── Step 5: FFmpeg assembly ──────────────────────────────────────────────────

def _get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [FFMPEG_EXE, "-i", str(audio_path)],
        capture_output=True, text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError("Could not determine audio duration")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _build_caption_chunks(boundaries: list[dict], chunk_size: int = 4) -> list[dict]:
    chunks = []
    for i in range(0, len(boundaries), chunk_size):
        group = boundaries[i: i + chunk_size]
        chunks.append({
            "text":  " ".join(w["word"] for w in group),
            "start": group[0]["start"],
            "end":   group[-1]["end"],
        })
    return chunks


def assemble_video(
    clip_paths: list[Path],
    audio_path: Path,
    word_boundaries: list[dict],
    script_data: dict,
    slug: str,
) -> Path:
    print("[5/5] Assembling final video...")

    audio_duration = _get_audio_duration(audio_path)
    out_path = OUTPUT_DIR / f"{slug}.mp4"

    # Build FFmpeg concat + stretch to match audio
    # Each clip is 5s; total clip duration = 5 * n_clips
    n_clips = len(clip_paths)
    clip_duration = 5.0
    total_clip_duration = n_clips * clip_duration

    # Write concat list
    concat_file = OUTPUT_DIR / f"{slug}_concat.txt"
    with open(concat_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp.resolve()}'\n")

    # Step A: Concat all clips into a silent video
    silent_path = OUTPUT_DIR / f"{slug}_silent.mp4"
    subprocess.run(
        [FFMPEG_EXE, "-y",
         "-f", "concat", "-safe", "0",
         "-i", str(concat_file),
         "-c", "copy",
         str(silent_path)],
        check=True, capture_output=True,
    )

    # Step B: Stretch/loop silent video to match audio duration
    stretched_path = OUTPUT_DIR / f"{slug}_stretched.mp4"
    if total_clip_duration < audio_duration:
        # Need to stretch — slow down video proportionally
        speed_factor = total_clip_duration / audio_duration  # < 1 = slow down
        pts_factor = 1.0 / speed_factor
        subprocess.run(
            [FFMPEG_EXE, "-y",
             "-i", str(silent_path),
             "-vf", f"setpts={pts_factor:.4f}*PTS",
             "-an",
             str(stretched_path)],
            check=True, capture_output=True,
        )
    else:
        # Trim to audio duration
        subprocess.run(
            [FFMPEG_EXE, "-y",
             "-i", str(silent_path),
             "-t", str(audio_duration),
             "-c", "copy",
             str(stretched_path)],
            check=True, capture_output=True,
        )

    # Step C: Build word-by-word karaoke captions (ASS format)
    def _ass_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        cs = int((s % 1) * 100)
        return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"

    # Find a good font — prefer Impact (TikTok style), fall back to Helvetica
    caption_font = "Impact"
    for fp in ["/Library/Fonts/Impact.ttf", "/System/Library/Fonts/Impact.ttf"]:
        if os.path.exists(fp):
            caption_font = "Impact"
            break
    else:
        caption_font = "Helvetica"

    ass_path = OUTPUT_DIR / f"{slug}.ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write(f"PlayResX: {VIDEO_W}\nPlayResY: {VIDEO_H}\nScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # White bold text, thick black outline, bottom-center at MarginV=160
        f.write(f"Style: Default,{caption_font},72,&H00FFFFFF,&H00FFFFFF,"
                f"&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,4,0,2,30,30,160,1\n")
        # Highlighted word style — yellow
        f.write(f"Style: Highlight,{caption_font},72,&H0000FFFF,&H0000FFFF,"
                f"&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,4,0,2,30,30,160,1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        # Group words into lines of 3 words max (karaoke per line)
        if word_boundaries:
            i = 0
            while i < len(word_boundaries):
                group = word_boundaries[i: i + 3]
                line_start = group[0]["start"]
                line_end   = group[-1]["end"]

                # Build karaoke line: each word tagged with its duration in centiseconds
                parts = []
                for w in group:
                    dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                    parts.append(f"{{\\k{dur_cs}}}{w['word'].upper()}")

                line_text = " ".join(parts)
                f.write(f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},"
                        f"Default,,0,0,0,,{line_text}\n")
                i += 3

    # Step D: Mux audio + burn ASS subtitles
    srt_path = ass_path  # reuse variable for the filter path

    subprocess.run(
        [FFMPEG_EXE, "-y",
         "-i", str(stretched_path),
         "-i", str(audio_path),
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-vf", (
             f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
             f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2,"
             f"ass={ass_path}"
         ),
         "-c:a", "aac", "-b:a", "192k",
         "-shortest",
         str(out_path)],
        check=True, capture_output=True,
    )

    # Cleanup temp files
    silent_path.unlink(missing_ok=True)
    stretched_path.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)

    print(f"\n✅ Video ready: {out_path}")
    return out_path


# ─── Main entry points ────────────────────────────────────────────────────────

def generate_video(topic: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"

    script_data          = generate_script(topic)
    audio_path, words    = generate_tts(script_data["script"], slug)
    image_urls           = generate_images(script_data["visual_prompts"], slug)
    clip_paths           = animate_images(image_urls, slug)
    video_path           = assemble_video(clip_paths, audio_path, words, script_data, slug)

    print(f"\n📋 Caption:\n{script_data['tiktok_caption']}")
    return video_path


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "octopuses have three hearts and blue blood"
    generate_video(topic)
