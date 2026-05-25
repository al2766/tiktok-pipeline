"""
@provenweird TikTok Video Pipeline — v3
Input:  topic string
Output: MP4 file in ./output/

Pipeline:
  1. Claude  → 12-sentence script + per-sentence visual prompts
  2. ElevenLabs → natural voice audio + word-level timestamps
  3. FAL AI (FLUX) → 12 HD 9:16 images, parallelised, cached
  4. Animation → hybrid: Kling for hero scenes, FFmpeg zoompan for the rest
  5. FFmpeg  → concat clips + mux audio + burn animated captions

Config env vars (all optional):
  VIDEO_MODE          = hybrid | full_kling | no_kling  (default: hybrid)
  HERO_SCENE_COUNT    = 3 or 4  (default: 4)
  HERO_SCENE_INDICES  = "0,5,10,11"  (overrides auto-selection)
  ENABLE_ASSET_CACHE  = true | false  (default: true)
  ENABLE_KLING_RESUME = true | false  (default: true)
"""

import os, json, re, time, subprocess, base64, gc, requests, textwrap, asyncio, hashlib, shutil
import jwt
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import fal_client
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY   = os.environ["ANTHROPIC_API_KEY"]
ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
KLING_ACCESS_KEY    = os.environ["KLING_ACCESS_KEY"]
KLING_SECRET_KEY    = os.environ["KLING_SECRET_KEY"]
FAL_KEY             = os.environ["FAL_KEY"]
os.environ["FAL_KEY"] = FAL_KEY

import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_W, VIDEO_H       = 1080, 1920
ELEVENLABS_VOICE_ID    = "nPczCjzI2devNBz1zQrb"  # Brian — deep, natural, male

# ─── Pipeline Config ──────────────────────────────────────────────────────────

VIDEO_MODE          = os.environ.get("VIDEO_MODE", "hybrid")           # hybrid | full_kling | no_kling
HERO_SCENE_COUNT    = int(os.environ.get("HERO_SCENE_COUNT", "4"))     # 3 or 4
_hero_env           = os.environ.get("HERO_SCENE_INDICES", "")
HERO_SCENE_INDICES  = [int(x) for x in _hero_env.split(",") if x.strip()] if _hero_env else None
ENABLE_ASSET_CACHE  = os.environ.get("ENABLE_ASSET_CACHE", "true").lower() == "true"
ENABLE_KLING_RESUME = os.environ.get("ENABLE_KLING_RESUME", "true").lower() == "true"

# ─── Asset Cache ──────────────────────────────────────────────────────────────
# Ephemeral (/tmp) — survives within one Render instance lifetime, not across deploys.
# Still valuable: retrying a crashed job reuses all completed assets.

CACHE_DIR = Path("/tmp/provenweird_cache")
for _sub in ["scripts", "images", "voice", "kling"]:
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


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

    if ENABLE_ASSET_CACHE:
        cache_path = CACHE_DIR / "scripts" / f"{_hash(topic)}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            print(f"   Script reused from cache.")
            return data

    system = """You write scripts for @provenweird — a faceless science facts TikTok. Your scripts are engineered to keep 65%+ of viewers past the 3-second mark. The algorithm punishes weak hooks and rewards completion.

VOICE: Tom Scott meets dry stand-up. Confident, slightly sardonic, conversational authority. NOT excited. NOT a textbook. Sound like a brilliant person who finds most explanations inadequate and is going to fix that — quickly.

CHOOSE THE BEST FORMULA for this topic and apply it:
- Formula A (PARADOX REVEAL): State the impossible-sounding fact upfront. Best for counterintuitive surface facts.
- Formula B (CURIOSITY GAP): Open the loop, don't answer it. Best when the MECHANISM is more interesting than the fact.
- Formula C (MYTH-BUST): "You've been told X. That's wrong." Best when a popular belief is false.
- Formula D (STAKES-FIRST): Open with the consequence affecting the viewer directly. Best for biology/psychology.
- Formula E (EXISTENTIAL PIVOT): Lead with the philosophical QUESTION the fact raises — not the fact itself. The science is the evidence, not the hook. Best when the fact implies something about identity, consciousness, existence, or time.
- Formula F (SCENARIO GAME): Put the viewer INSIDE a hypothetical scenario. The fact answers what happens next. Opens with "What if you were..." The science is the answer to a question they're now personally invested in.
- Formula G (SCALE REVEAL): Use a geographic or population-scale comparison to make an abstract fact viscerally real.
- Formula H (FAMOUS PROXY): Use a celebrity, popular book/franchise, or already-loved concept as the vehicle. The viewer already has emotional connection to the subject.

STRUCTURE (exactly 12 sentences — minimum 60 seconds for TikTok Creator Rewards):
1.  HOOK — The scroll-stopper. State the paradox, gap, myth, stakes, or scenario. NEVER "Did you know". Max 15 words.
2-3. DESTABILISE — Confirm the hook is real. Add a layer that makes it even stranger.
4.  SPECIFIC NUMBER — One exact measurement or statistic. This anchors credibility.
5-9. MECHANISM — Explain WHY, fast. Max 12 words per sentence. Plain language, no passive voice.
10. HUMAN COMPARISON — Scale it to something absurd and relatable. The shareable moment.
11. TWIST — The most unexpected implication. The thing they didn't see coming.
12. KICKER — Dry callback to the hook. Reframes everything. The line people screenshot.

NON-NEGOTIABLE RULES:
- NEVER "Did you know", "Today we're", passive openers
- NEVER end on a question to the viewer
- Exactly one specific number — not zero, not two
- Total: 90-130 words (60-70 seconds at natural pace)
- Second-person where possible ("your", "you", "imagine")
- Dry = confident + specific. NOT dismissive. The hook must make people feel smart, not targeted.
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
            data["script"] = " ".join(data["sentences"])
            print(f"   Hook: {data['hook']}")
            if ENABLE_ASSET_CACHE:
                cache_path.write_text(json.dumps(data))
            return data
        except json.JSONDecodeError:
            time.sleep(2)
    raise RuntimeError("Script generation failed after 3 attempts")


# ─── Step 2: TTS with word timestamps ─────────────────────────────────────────

def _whisper_word_boundaries(audio_path: Path) -> list[dict]:
    from faster_whisper import WhisperModel
    model_dir = Path(__file__).parent / "whisper_models"
    model_dir.mkdir(exist_ok=True)
    model = WhisperModel("small", device="cpu", compute_type="int8",
                         download_root=str(model_dir))
    segments_raw, _ = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)
    boundaries = []
    for seg in segments_raw:
        for w in seg.words:
            boundaries.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    del model
    return boundaries


def _elevenlabs_tts(script: str, audio_path: Path) -> list[dict] | None:
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/with-timestamps",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.85, "style": 0.2},
        },
        timeout=60,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    audio_path.write_bytes(base64.b64decode(data["audio_base64"]))
    chars = data.get("alignment", {})
    if not chars:
        return None
    char_starts = chars.get("character_start_times_seconds", [])
    char_ends   = chars.get("character_end_times_seconds", [])
    char_list   = chars.get("characters", [])
    boundaries = []
    i, n = 0, len(char_list)
    while i < n:
        while i < n and char_list[i] == " ":
            i += 1
        if i >= n:
            break
        j = i
        while j < n and char_list[j] != " ":
            j += 1
        word = "".join(char_list[i:j])
        boundaries.append({"word": word, "start": char_starts[i], "end": char_ends[j - 1]})
        i = j
    print(f"   {len(boundaries)} word boundaries, audio saved.")
    return boundaries


def _edge_tts_generate(script: str, audio_path: Path):
    async def _run():
        import edge_tts
        c = edge_tts.Communicate(script, "en-GB-RyanNeural")
        await c.save(str(audio_path))
    asyncio.run(_run())


def generate_tts(script: str, slug: str) -> tuple[Path, list[dict]]:
    audio_path = OUTPUT_DIR / f"{slug}_audio.mp3"

    if ENABLE_ASSET_CACHE:
        cache_key   = _hash(script)
        audio_cache = CACHE_DIR / "voice" / f"{cache_key}.mp3"
        bound_cache = CACHE_DIR / "voice" / f"{cache_key}.json"
        if audio_cache.exists() and bound_cache.exists():
            shutil.copy2(audio_cache, audio_path)
            boundaries = json.loads(bound_cache.read_text())
            print(f"[2/5] Voice reused from cache ({len(boundaries)} boundaries).")
            return audio_path, boundaries

    print("[2/5] Generating voice with ElevenLabs...")
    boundaries = _elevenlabs_tts(script, audio_path)
    if boundaries is None:
        print("   ElevenLabs unavailable — falling back to edge-tts + Whisper timestamps")
        _edge_tts_generate(script, audio_path)
        boundaries = _whisper_word_boundaries(audio_path)

    if ENABLE_ASSET_CACHE:
        shutil.copy2(audio_path, audio_cache)
        bound_cache.write_text(json.dumps(boundaries))

    return audio_path, boundaries


# ─── Step 3: FAL AI images — parallelised + cached ────────────────────────────

def generate_images(visual_prompts: list[str], slug: str) -> tuple[list[Path], dict]:
    """Returns (image_paths, cost_info). Runs in parallel with caching."""
    n = len(visual_prompts)
    print(f"[3/5] Generating {n} images via FAL FLUX...")
    img_dir = OUTPUT_DIR / f"{slug}_images"
    img_dir.mkdir(exist_ok=True)
    image_paths: list[Path | None] = [None] * n
    cost_info = {"generated": 0, "reused": 0}

    def _one(i: int, prompt: str) -> tuple[int, Path, str]:
        cache_key  = _hash(prompt)
        cache_path = CACHE_DIR / "images" / f"{cache_key}.jpg"
        img_path   = img_dir / f"img_{i:02d}.jpg"

        if ENABLE_ASSET_CACHE and cache_path.exists():
            shutil.copy2(cache_path, img_path)
            return i, img_path, "reused"

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
        url      = result["images"][0]["url"]
        img_data = requests.get(url, timeout=60).content
        img_path.write_bytes(img_data)
        if ENABLE_ASSET_CACHE:
            cache_path.write_bytes(img_data)
        return i, img_path, "generated"

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_one, i, p): i for i, p in enumerate(visual_prompts)}
        for fut in as_completed(futures):
            i, path, status = fut.result()
            image_paths[i] = path
            cost_info[status] += 1
            print(f"   ✓ img_{i:02d} ({status})")

    return image_paths, cost_info


# ─── Step 4a: Kling image-to-video (hero scenes) ──────────────────────────────

def _kling_jwt() -> str:
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256",
                      headers={"alg": "HS256", "typ": "JWT"})


def _kling_submit(image_path: Path, motion_prompt: str) -> str:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "model_name": "kling-v1",
        "image":       img_b64,
        "prompt":      motion_prompt,
        "duration":    "5",
        "mode":        "std",
        "aspect_ratio": "9:16",
    }
    for attempt, wait in enumerate([0, 30, 60, 120]):
        if wait:
            print(f"   Kling 429 — waiting {wait}s before retry {attempt}...")
            time.sleep(wait)
        token = _kling_jwt()
        resp = requests.post(
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
    raise RuntimeError("Kling submit failed after 4 attempts (persistent 429 rate limit)")


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
        task   = data["data"]
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

KLING_CONCURRENCY = 3


# ─── Step 4b: FFmpeg zoompan motion (non-hero scenes) ─────────────────────────

# 12-position preset cycle — varied so sequential scenes don't look identical
_FFMPEG_MOTION_CYCLE = [
    "slow_zoom_in", "pan_left",     "slow_zoom_out", "drift_up",
    "pan_right",    "slow_zoom_in", "pan_left",      "slow_zoom_out",
    "drift_up",     "pan_right",    "slow_zoom_in",  "slow_zoom_out",
]


def _motion_vf(preset: str, n_frames: int = 125) -> str:
    # Scale+crop to 1440x2560 first (gives 33% headroom for zoom/pan into 1080x1920 output)
    d  = n_frames
    sc = "scale=1440:2560:force_original_aspect_ratio=increase,crop=1440:2560"
    zp = f"d={d}:s=1080x1920:fps=25"
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"
    table = {
        "slow_zoom_in":  f"{sc},zoompan=z='min(zoom+0.002,1.25)':x='{cx}':y='{cy}':{zp}",
        "slow_zoom_out": f"{sc},zoompan=z='if(lte(zoom,1.0),1.25,max(1.01,zoom-0.002))':x='{cx}':y='{cy}':{zp}",
        "pan_left":      f"{sc},zoompan=z=1.2:x='if(lte(on,1),0,min(iw*(1-1/zoom),x+1.5))':y='{cy}':{zp}",
        "pan_right":     f"{sc},zoompan=z=1.2:x='if(lte(on,1),iw*(1-1/zoom),max(0,x-1.5))':y='{cy}':{zp}",
        "drift_up":      f"{sc},zoompan=z=1.2:x='{cx}':y='if(lte(on,1),ih*(1-1/zoom),max(0,y-1.5))':{zp}",
    }
    return table.get(preset, table["slow_zoom_in"])


def _ffmpeg_motion_clip(image_path: Path, clip_path: Path, preset: str, duration: float = 5.0) -> Path:
    fps      = 25
    n_frames = int(duration * fps)
    vf       = _motion_vf(preset, n_frames)
    cmd = [
        FFMPEG_EXE, "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg motion failed for {clip_path.name}: {result.stderr[-400:]}")
    return clip_path


# ─── Step 4: Animate (hybrid dispatcher) ──────────────────────────────────────

def _hero_indices(n_scenes: int) -> set[int]:
    if HERO_SCENE_INDICES:
        return {i for i in HERO_SCENE_INDICES if i < n_scenes}
    count = min(HERO_SCENE_COUNT, n_scenes)
    if count >= 4 and n_scenes >= 4:
        mid = n_scenes // 2
        return {0, mid, n_scenes - 2, n_scenes - 1}
    if count >= 3 and n_scenes >= 3:
        return {0, n_scenes - 2, n_scenes - 1}
    if count >= 2:
        return {0, n_scenes - 1}
    return {0}


def animate_images(image_paths: list[Path], slug: str) -> tuple[list[Path], dict]:
    """
    Returns (clip_paths, cost_info).
    cost_info keys: kling_generated, kling_reused, ffmpeg_clips, kling_units_used
    """
    n         = len(image_paths)
    heroes    = _hero_indices(n) if VIDEO_MODE == "hybrid" else (set(range(n)) if VIDEO_MODE == "full_kling" else set())
    ffmpeg_ct = n - len(heroes) if VIDEO_MODE == "hybrid" else (0 if VIDEO_MODE == "full_kling" else n)

    print(f"[4/5] Animating {n} images (mode={VIDEO_MODE}, kling={len(heroes)}, ffmpeg={ffmpeg_ct})...")
    if heroes:
        print(f"   Hero scenes (Kling): {sorted(heroes)}")

    clip_dir = OUTPUT_DIR / f"{slug}_clips"
    clip_dir.mkdir(exist_ok=True)

    cost = {"kling_generated": 0, "kling_reused": 0, "ffmpeg_clips": 0, "kling_units_used": 0.0}
    video_paths: list[Path | None] = [None] * n

    # ── Kling hero scenes ──────────────────────────────────────────────────
    kling_queue: list[tuple[int, str]] = []   # (scene_index, task_id)

    for i in sorted(heroes):
        img_path   = image_paths[i]
        clip_path  = clip_dir / f"clip_{i:02d}.mp4"
        motion     = MOTION_PROMPTS[i % len(MOTION_PROMPTS)]
        img_bytes  = img_path.read_bytes()
        cache_key  = _hash(f"{_hash_bytes(img_bytes)}:{motion}")
        cache_path = CACHE_DIR / "kling" / f"{cache_key}.mp4"

        if ENABLE_KLING_RESUME and clip_path.exists() and clip_path.stat().st_size > 10_000:
            print(f"   ✓ Clip {i+1} resumed (on disk)")
            video_paths[i] = clip_path
            cost["kling_reused"] += 1
            continue

        if ENABLE_ASSET_CACHE and cache_path.exists():
            shutil.copy2(cache_path, clip_path)
            print(f"   ✓ Clip {i+1} from cache (Kling)")
            video_paths[i] = clip_path
            cost["kling_reused"] += 1
            continue

        print(f"   Submitting Kling clip {i+1}/{n} (hero)...")
        task_id = _kling_submit(img_path, motion)
        kling_queue.append((i, task_id, clip_path, cache_key))

    # Poll Kling submissions in batches of KLING_CONCURRENCY
    for batch_start in range(0, len(kling_queue), KLING_CONCURRENCY):
        batch = kling_queue[batch_start: batch_start + KLING_CONCURRENCY]
        for i, task_id, clip_path, cache_key in batch:
            print(f"   Waiting for clip {i+1} (task {task_id})...")
            video_url  = _kling_poll(task_id)
            r          = requests.get(video_url, timeout=120)
            r.raise_for_status()
            clip_path.write_bytes(r.content)
            if ENABLE_ASSET_CACHE:
                (CACHE_DIR / "kling" / f"{cache_key}.mp4").write_bytes(r.content)
            print(f"   ✓ Clip {i+1} saved ({len(r.content) // 1024}KB)")
            video_paths[i] = clip_path
            cost["kling_generated"] += 1
            cost["kling_units_used"] += 1.5

    # ── FFmpeg non-hero scenes ─────────────────────────────────────────────
    for i, img_path in enumerate(image_paths):
        if video_paths[i] is not None:
            continue
        clip_path = clip_dir / f"clip_{i:02d}.mp4"
        preset    = _FFMPEG_MOTION_CYCLE[i % len(_FFMPEG_MOTION_CYCLE)]
        print(f"   FFmpeg clip {i+1}/{n} ({preset})...")
        _ffmpeg_motion_clip(img_path, clip_path, preset)
        print(f"   ✓ Clip {i+1} ready")
        video_paths[i] = clip_path
        cost["ffmpeg_clips"] += 1

    return video_paths, cost


# ─── Step 5: FFmpeg assembly ───────────────────────────────────────────────────

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
            "words":  [w["word"] for w in group],
            "start":  group[0]["start"],
            "end":    group[-1]["end"],
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

    audio_duration      = _get_audio_duration(audio_path)
    out_path            = OUTPUT_DIR / f"{slug}.mp4"
    n_clips             = len(clip_paths)
    clip_duration       = 5.0
    total_clip_duration = n_clips * clip_duration

    concat_file = OUTPUT_DIR / f"{slug}_concat.txt"
    with open(concat_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp.resolve()}'\n")

    silent_path = OUTPUT_DIR / f"{slug}_silent.mp4"
    subprocess.run(
        [FFMPEG_EXE, "-y",
         "-f", "concat", "-safe", "0",
         "-i", str(concat_file),
         "-c", "copy",
         str(silent_path)],
        check=True, capture_output=True,
    )

    stretched_path = OUTPUT_DIR / f"{slug}_stretched.mp4"
    if total_clip_duration < audio_duration:
        speed_factor = total_clip_duration / audio_duration
        pts_factor   = 1.0 / speed_factor
        subprocess.run(
            [FFMPEG_EXE, "-y",
             "-i", str(silent_path),
             "-vf", f"setpts={pts_factor:.4f}*PTS",
             "-an",
             str(stretched_path)],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            [FFMPEG_EXE, "-y",
             "-i", str(silent_path),
             "-t", str(audio_duration),
             "-c", "copy",
             str(stretched_path)],
            check=True, capture_output=True,
        )

    def _ass_time(t: float) -> str:
        h  = int(t // 3600)
        m  = int((t % 3600) // 60)
        s  = t % 60
        cs = int((s % 1) * 100)
        return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"

    caption_font = "Impact"
    for fp in ["/Library/Fonts/Impact.ttf", "/System/Library/Fonts/Impact.ttf",
               "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"]:
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
        f.write(f"Style: Default,{caption_font},72,&H00FFFFFF,&H00FFFFFF,"
                f"&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,4,0,2,30,30,160,1\n")
        f.write(f"Style: Highlight,{caption_font},72,&H0000FFFF,&H0000FFFF,"
                f"&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,4,0,2,30,30,160,1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        if word_boundaries:
            i = 0
            while i < len(word_boundaries):
                group      = word_boundaries[i: i + 3]
                line_start = group[0]["start"]
                line_end   = group[-1]["end"]
                parts = []
                for w in group:
                    dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                    parts.append(f"{{\\k{dur_cs}}}{w['word'].upper()}")
                line_text = " ".join(parts)
                f.write(f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},"
                        f"Default,,0,0,0,,{line_text}\n")
                i += 3

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

    silent_path.unlink(missing_ok=True)
    stretched_path.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)
    ass_path.unlink(missing_ok=True)

    print(f"\n✅ Video ready: {out_path}")
    return out_path


# ─── Cost summary ─────────────────────────────────────────────────────────────

_KLING_UNIT_COST  = 0.098   # USD per unit (std mode)
_FAL_IMAGE_COST   = 0.003   # USD per FLUX schnell image
_ELEVENLABS_COST  = 0.05    # USD approximate per video (Starter plan)
_CLAUDE_COST      = 0.02    # USD approximate per script


def print_cost_summary(
    img_cost:  dict,
    clip_cost: dict,
    audio_duration: float,
    video_duration: float,
    voice_reused: bool = False,
    script_reused: bool = False,
):
    kling_total = clip_cost["kling_generated"] + clip_cost["kling_reused"]
    fal_gen     = img_cost["generated"]
    fal_reused  = img_cost["reused"]

    kling_usd   = clip_cost["kling_units_used"] * _KLING_UNIT_COST
    fal_usd     = fal_gen * _FAL_IMAGE_COST
    el_usd      = 0.0 if voice_reused else _ELEVENLABS_COST
    claude_usd  = 0.0 if script_reused else _CLAUDE_COST
    total_usd   = kling_usd + fal_usd + el_usd + claude_usd

    generated_secs = kling_total * 5.0
    waste_pct      = max(0, (generated_secs - video_duration) / generated_secs * 100) if generated_secs else 0

    print("\n" + "═" * 50)
    print("📊 COST SUMMARY")
    print(f"   Mode:          {VIDEO_MODE}")
    print(f"   FAL images:    generated={fal_gen}, reused={fal_reused}  (~${fal_usd:.3f})")
    print(f"   Kling clips:   generated={clip_cost['kling_generated']}, reused={clip_cost['kling_reused']}  "
          f"(~{clip_cost['kling_units_used']:.1f} units / ~${kling_usd:.2f})")
    print(f"   FFmpeg clips:  {clip_cost['ffmpeg_clips']}  ($0.00)")
    print(f"   ElevenLabs:    {'reused' if voice_reused else f'~${el_usd:.2f}'}")
    print(f"   Claude:        {'reused' if script_reused else f'~${claude_usd:.2f}'}")
    print(f"   ─────────────────────────────────────")
    print(f"   TOTAL:         ~${total_usd:.2f}")
    print(f"   Video:         {video_duration:.1f}s final  |  {generated_secs:.0f}s Kling generated  |  {waste_pct:.0f}% waste")
    print("═" * 50 + "\n")

    return {
        "total_usd":       round(total_usd, 3),
        "kling_units":     clip_cost["kling_units_used"],
        "fal_generated":   fal_gen,
        "fal_reused":      fal_reused,
        "kling_generated": clip_cost["kling_generated"],
        "kling_reused":    clip_cost["kling_reused"],
        "ffmpeg_clips":    clip_cost["ffmpeg_clips"],
        "video_seconds":   round(video_duration, 1),
    }


# ─── Main entry points ────────────────────────────────────────────────────────

def generate_video(topic: str) -> tuple[Path, dict]:
    """Returns (video_path, cost_summary_dict)."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"

    script_data        = generate_script(topic)
    audio_path, words  = generate_tts(script_data["script"], slug)
    image_paths, i_cost = generate_images(script_data["visual_prompts"], slug)
    clip_paths, c_cost  = animate_images(image_paths, slug)
    video_path         = assemble_video(clip_paths, audio_path, words, script_data, slug)

    audio_dur  = _get_audio_duration(audio_path)
    cost_summary = print_cost_summary(i_cost, c_cost, audio_dur, audio_dur)

    print(f"\n📋 Caption:\n{script_data['tiktok_caption']}")
    return video_path, cost_summary


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "octopuses have three hearts and blue blood"
    generate_video(topic)
