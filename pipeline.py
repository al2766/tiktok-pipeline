"""
@provenweird TikTok Video Pipeline
Input:  topic string
Output: single MP4 in ./output/ — no intermediates kept
Steps:  1. Claude (cached system prompt) → 12-sentence script + visual prompts
        2. ElevenLabs → voice MP3 + word timestamps
        3. FAL FLUX → 12 × 9:16 images
        4. Kling AI (direct API, JWT auth) → animate each image into 5s clip
        5. FFmpeg → concat clips → stretch → karaoke captions → mux audio
        6. Google Drive upload (optional)
"""

import os, json, re, time, subprocess, base64, requests, asyncio, shutil
import jwt  # PyJWT
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
KLING_ACCESS_KEY   = os.environ["KLING_ACCESS_KEY"]
KLING_SECRET_KEY   = os.environ["KLING_SECRET_KEY"]
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

VIDEO_W, VIDEO_H    = 1080, 1920
ELEVENLABS_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # Brian — deep, natural

_SCRIPT_SYSTEM = """You write scripts for @provenweird — a faceless science facts TikTok built to stop scrolling cold.

VOICE: Dry, slightly sarcastic, mysteriously confident. You know something the viewer doesn't — and you find their ignorance mildly amusing, but you're on their side. You're the cool scientist at the back of the bar who just told you something that ruins how you see the world forever. Deliver facts like they're slightly scandalous secrets someone forgot to classify. Never excited. Never teacher-ish. The sarcasm is aimed at the universe ("of course it works like that") — never at the viewer.

SHOCK FIRST: Every script must contain at least one fact that makes a person stop mid-scroll and think "that can't be real." Lead with the disturbing or destabilising implication — not the polite explanation.

MYSTERY: Leave a gap. Drop one fact and let it hang in the air a half-beat longer than it should. The viewer should feel like they're being let in on something hidden.

CHOOSE THE BEST FORMULA for this topic:
- Formula A (PARADOX REVEAL): The fact sounds physically impossible. State it deadpan upfront.
- Formula B (CURIOSITY GAP): Open the loop with something wrong or missing. Don't answer it yet.
- Formula C (MYTH-BUST): "Everyone thinks X. That's not even close." Best for widely held false beliefs.
- Formula D (STAKES-FIRST): Open with the consequence affecting the viewer's body or mind directly.
- Formula E (EXISTENTIAL PIVOT): The fact reframes something about identity, time, consciousness, or reality. Open with the question it raises, not the fact.

STRUCTURE (exactly 12 sentences, minimum 60 seconds):
1.  HOOK — Scroll-stopper. Shocking or destabilising. NEVER "Did you know". Max 15 words.
2.  CONFIRM — Make it real. One sentence that proves the hook isn't exaggerating.
3.  DEEPEN — A second angle on the same fact that makes it stranger. Not a continuation — a new disturbing layer.
4.  NUMBER — One exact, unrounded measurement or statistic. The credibility anchor.
5-8. MECHANISM — How and why, in punchy sentences of max 10 words each. Somewhere here: one sentence the viewer could not have predicted. A fact that shifts the ground. Keep the sarcastic undertone — "turns out your body just... does this."
9.  CONSEQUENCE — What this actually means in the real world. Specific. Not vague.
10. SCALE — Make the viewer feel it physically. Not a clever comparison — a visceral one.
11. TWIST — The implication no one mentions. The thing they'll repeat tonight.
12. KICKER — A single true statement that reframes the hook. Delivered with quiet confidence. NOT a joke. NOT a pop culture reference. NOT a meta-comment. Just the fact, stated like it's obvious — because now it is.

NON-NEGOTIABLE RULES:
- NEVER open with "Did you know", "Today we're", "It has been", or passive openers
- NEVER end on a question to the viewer
- NEVER use a joke, punchline, or "worse branding" style comparison in the kicker
- Exactly one number — not zero, not two
- Each sentence should feel like it could stop a scroll on its own
- 90-130 words total (60-70 seconds at natural pace)
- Second-person where it fits ("your", "you", "imagine")
- Halal content only

Return ONLY valid JSON, no markdown, no backticks:
{"hook":"first sentence only","sentences":["exactly 12 strings"],"visual_prompts":["exactly 12 cinematic image prompts — photorealistic or hyper-detailed illustration, 9:16 vertical, vivid and dramatic, directly illustrates the sentence, no text in image"],"tiktok_caption":"punchy 1-line caption with the wildest fact + 5 hashtags"}"""


# ─── Step 1: Script ───────────────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    print(f"[1/5] Generating script: {topic}")
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
    print("[2/5] Generating voice...")
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
    """Generate images via FAL FLUX. Returns local paths (Kling receives base64, not URLs)."""
    print(f"[3/5] Generating {len(visual_prompts)} images...")
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
        url      = resp.json()["images"][0]["url"]
        img_path = img_dir / f"{i:02d}.jpg"
        img_path.write_bytes(requests.get(url, timeout=60).content)
        print(f"   {i+1}/{len(visual_prompts)} ✓")
        paths.append(img_path)
    return paths


# ─── Step 4: Kling image-to-video (direct API, JWT auth) ─────────────────────

def _kling_jwt() -> str:
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256",
                      headers={"alg": "HS256", "typ": "JWT"})


def _kling_submit(image_path: Path, motion_prompt: str) -> str:
    """Submit one image-to-video task via base64. Returns task_id."""
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "model_name": "kling-v1",
        "image":      img_b64,
        "prompt":     motion_prompt,
        "duration":   "5",
        "mode":       "std",
        "aspect_ratio": "9:16",
    }
    for attempt, wait in enumerate([0, 30, 60, 120]):
        if wait:
            print(f"   Kling 429 — waiting {wait}s before retry {attempt}...")
            time.sleep(wait)
        token = _kling_jwt()
        resp  = requests.post(
            "https://api.klingai.com/v1/videos/image2video",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 429:
            body = resp.json() if resp.content else {}
            if body.get("code") == 1102:
                raise RuntimeError("Kling account balance is empty — top up credits at klingai.com")
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling submit failed: {data}")
        task_id = data["data"]["task_id"]
        print(f"   Submitted task {task_id}")
        return task_id
    raise RuntimeError("Kling submit failed after 4 attempts (persistent 429)")


def _kling_poll(task_id: str, timeout: int = 300) -> str:
    """Poll until task is done. Returns video URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(10)
        token = _kling_jwt()
        resp  = requests.get(
            f"https://api.klingai.com/v1/videos/image2video/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data   = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling poll error: {data}")
        task   = data["data"]
        status = task.get("task_status", "")
        if status == "succeed":
            return task["task_result"]["videos"][0]["url"]
        if status == "failed":
            raise RuntimeError(f"Kling task failed: {task.get('task_status_msg')}")
        print(f"   Kling {task_id}: {status}...")
    raise RuntimeError(f"Kling task {task_id} timed out after {timeout}s")


_MOTION_PROMPTS = [
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

KLING_CONCURRENCY = 3


def animate_images(image_paths: list[Path], slug: str) -> list[Path]:
    """Animate images with Kling (direct API). 3 concurrent, 300s per-clip timeout."""
    print(f"[4/5] Animating {len(image_paths)} images with Kling...")
    clip_dir = OUTPUT_DIR / f"{slug}_clips"
    clip_dir.mkdir(exist_ok=True)

    video_paths = []
    for batch_start in range(0, len(image_paths), KLING_CONCURRENCY):
        batch         = image_paths[batch_start: batch_start + KLING_CONCURRENCY]
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        task_ids = []
        for i, img_path in zip(batch_indices, batch):
            motion  = _MOTION_PROMPTS[i % len(_MOTION_PROMPTS)]
            print(f"   Submitting clip {i+1}/{len(image_paths)}...")
            task_id = _kling_submit(img_path, motion)
            task_ids.append((i, task_id))
            time.sleep(1)

        for i, task_id in task_ids:
            print(f"   Waiting for clip {i+1}/{len(image_paths)} (task {task_id})...")
            video_url  = _kling_poll(task_id)
            clip_path  = clip_dir / f"clip_{i:02d}.mp4"
            r = requests.get(video_url, timeout=120)
            r.raise_for_status()
            clip_path.write_bytes(r.content)
            print(f"   ✓ Clip {i+1} saved ({len(r.content)//1024}KB)")
            video_paths.append(clip_path)

    return video_paths


# ─── Step 5: Assemble ─────────────────────────────────────────────────────────

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


def _kenburns_clips(image_paths: list[Path], tmp: Path) -> list[Path]:
    """Ken Burns zoompan fallback — used when Kling is unavailable."""
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
        print(f"   Ken Burns clip {i+1}/{len(image_paths)} ✓")
    return clip_paths


def assemble_clips(
    clip_paths: list[Path],
    audio_path: Path,
    boundaries: list[dict],
    slug: str,
    cleanup_images: list[Path] | None = None,
) -> Path:
    """Concat clips + mux audio + burn karaoke captions → final MP4."""
    print("[5/5] Assembling final video...")
    tmp      = clip_paths[0].parent
    out_path = OUTPUT_DIR / f"{slug}.mp4"

    concat_list = tmp / "clips.txt"
    with open(concat_list, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c.resolve()}'\n")

    silent = tmp / "silent.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(silent)],
        check=True, capture_output=True,
    )

    audio_dur = _audio_duration(audio_path)

    def _clip_dur(p: Path) -> float:
        r = subprocess.run([FFMPEG, "-i", str(p)], capture_output=True, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
        if not m:
            return 6.0
        h, mn, s = m.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)

    total_dur = sum(_clip_dur(c) for c in clip_paths)

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

    shutil.rmtree(tmp, ignore_errors=True)
    if cleanup_images:
        img_dir = cleanup_images[0].parent if cleanup_images else None
        for p in cleanup_images:
            p.unlink(missing_ok=True)
        if img_dir and img_dir.exists():
            shutil.rmtree(img_dir, ignore_errors=True)
    audio_path.unlink(missing_ok=True)

    print(f"✅ {out_path.name}")
    return out_path


def assemble_video(image_paths: list[Path], audio_path: Path, boundaries: list[dict], slug: str) -> Path:
    """Ken Burns fallback path."""
    tmp = OUTPUT_DIR / f"{slug}_tmp"
    tmp.mkdir(exist_ok=True)
    clip_paths = _kenburns_clips(image_paths, tmp)
    return assemble_clips(clip_paths, audio_path, boundaries, slug, cleanup_images=image_paths)


# ─── Google Drive upload (optional) ──────────────────────────────────────────

def upload_to_drive(video_path: Path) -> str | None:
    """Upload video to Google Drive. Returns shareable view URL, or None if not configured."""
    creds_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    folder_id  = os.environ.get("GDRIVE_FOLDER_ID")
    if not creds_json or not folder_id:
        return None
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds   = Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        file_meta = {"name": video_path.name, "parents": [folder_id]}
        media     = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
        f         = service.files().create(body=file_meta, media_body=media, fields="id,webViewLink").execute()

        service.permissions().create(fileId=f["id"], body={"type": "anyone", "role": "reader"}).execute()

        url = f.get("webViewLink")
        print(f"   Drive: {url}")
        return url
    except Exception as e:
        print(f"   Drive upload failed (non-fatal): {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_video(topic: str) -> tuple[Path, str]:
    slug        = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"
    script      = generate_script(topic)
    audio, wds  = generate_tts(script["script"], slug)
    image_paths = generate_images(script["visual_prompts"], slug)
    clip_paths  = animate_images(image_paths, slug)
    video       = assemble_clips(clip_paths, audio, wds, slug, cleanup_images=image_paths)
    caption     = script.get("tiktok_caption", "")
    print(f"\nCaption: {caption}")
    return video, caption


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "honey never expires — found in 3000-year-old Egyptian tombs"
    generate_video(topic)
