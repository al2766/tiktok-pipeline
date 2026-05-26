"""
@provenweird TikTok Video Pipeline
Input:  topic string
Output: single MP4 in ./output/ — no intermediates kept
Steps:  1. Claude (cached system prompt) → 12-sentence script + visual prompts
        2. ElevenLabs → voice MP3 + word timestamps
        3. FAL FLUX → 12 × 9:16 images
        4. FFmpeg → Ken Burns per image → concat → stretch → captions → mux audio
        5. Google Drive upload (optional — requires GDRIVE_SERVICE_ACCOUNT_JSON + GDRIVE_FOLDER_ID env vars)
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

VOICE: Confident, precise, slightly sardonic. You state facts like they are self-evidently wild. NOT comedic. NOT a stand-up set. NOT a textbook. No jokes. No meta-comments about the content. No punchlines. The facts are stranger than any joke — let them land without embellishment.

CHOOSE THE BEST FORMULA for this topic:
- Formula A (PARADOX REVEAL): State the impossible-sounding fact upfront. Best for counterintuitive surface facts.
- Formula B (CURIOSITY GAP): Open the loop, don't answer it. Best when the MECHANISM is more interesting than the fact.
- Formula C (MYTH-BUST): "You've been told X. That's wrong." Best when a popular belief is false.
- Formula D (STAKES-FIRST): Open with the consequence affecting the viewer directly. Best for biology/psychology.
- Formula E (EXISTENTIAL PIVOT): Lead with the philosophical QUESTION the fact raises — not the fact itself. Best when the fact implies something about identity, consciousness, or time.

STRUCTURE (exactly 12 sentences, minimum 60 seconds of speech):
1.  HOOK — The scroll-stopper. NEVER "Did you know". Max 15 words.
2-3. DESTABILISE — Confirm the hook is real. Then take it somewhere the viewer didn't expect. Not a continuation — a new angle on the same fact.
4.  SPECIFIC NUMBER — One exact measurement or statistic. Not rounded. This is the credibility anchor.
5-8. MECHANISM — Explain WHY in short, punchy sentences. Max 10 words each. Somewhere in here, include one sentence that pivots unexpectedly — a fact the viewer couldn't have predicted from the previous sentence. Make them feel like the ground shifted.
9.  CONSEQUENCE — The real-world result of the mechanism. One sentence. Specific, not vague.
10. HUMAN COMPARISON — Scale it to something visceral and relatable. Not clever — visceral. The viewer should feel it in their body.
11. TWIST — The implication nobody mentioned. The thing the viewer will repeat to someone else tonight.
12. KICKER — A final true statement that reframes the whole thing. NOT a joke. NOT a comparison to pop culture. NOT a meta-comment. A genuine fact or implication that makes the hook land differently the second time.

NON-NEGOTIABLE RULES:
- NEVER open with "Did you know", "Today we're", "It has been", or any passive opener
- NEVER end on a question to the viewer
- NEVER use pop culture, book, or film references in the kicker
- NEVER write a joke or punchline — the strangeness of the fact IS the payoff
- Exactly one specific number — not zero, not two
- The viewer should not be able to predict the next sentence from the previous one
- Total script: 90-130 words (60-70 seconds at natural pace)
- Second-person where possible ("your", "you", "imagine")
- Halal content only

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

def generate_images(visual_prompts: list[str], slug: str) -> tuple[list[Path], list[str]]:
    """Returns (local_paths, fal_cdn_urls) — URLs are passed to AI animation step."""
    print(f"[3/5] Generating {len(visual_prompts)} images...")
    img_dir = OUTPUT_DIR / f"{slug}_img"
    img_dir.mkdir(exist_ok=True)
    paths, urls = [], []
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
        urls.append(url)
    return paths, urls


# ─── Step 4: AI animation via MiniMax (FAL queue) ─────────────────────────────

_MOTION_STYLES = [
    "slow cinematic zoom in, dramatic atmosphere, smooth motion",
    "gentle camera drift left, ethereal lighting, fluid",
    "slow pull back revealing scale, epic atmosphere",
    "subtle camera movement upward, intense and cinematic",
    "slow zoom out, atmospheric depth, dramatic reveal",
    "gentle rightward pan, vivid and dynamic, cinematic",
]

_FAL_QUEUE = "https://queue.fal.run"
_MINIMAX_MODEL = "fal-ai/minimax/video-01-live"


def animate_images_ai(image_urls: list[str], slug: str) -> list[Path]:
    """Submit all images to MiniMax simultaneously via FAL queue. Falls back to Ken Burns on error."""
    print(f"[4/5] Animating {len(image_urls)} images with MiniMax AI (parallel)...")
    tmp = OUTPUT_DIR / f"{slug}_tmp"
    tmp.mkdir(exist_ok=True)

    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}

    # Submit all requests at once
    request_ids: list[str] = []
    for i, img_url in enumerate(image_urls):
        motion = _MOTION_STYLES[i % len(_MOTION_STYLES)]
        try:
            r = requests.post(
                f"{_FAL_QUEUE}/{_MINIMAX_MODEL}",
                headers=headers,
                json={"image_url": img_url, "prompt": motion},
                timeout=30,
            )
            r.raise_for_status()
            request_ids.append(r.json()["request_id"])
            print(f"   Queued {i+1}/{len(image_urls)}")
        except Exception as e:
            raise RuntimeError(f"MiniMax submit failed for clip {i+1}: {e}")

    # Poll all in parallel until every clip is done
    clips: list[Path | None] = [None] * len(request_ids)
    pending   = set(range(len(request_ids)))
    deadline  = time.time() + 600  # 10-minute hard timeout

    while pending:
        if time.time() > deadline:
            raise RuntimeError("MiniMax animation timed out after 10 minutes")
        time.sleep(8)
        for i in list(pending):
            try:
                st = requests.get(
                    f"{_FAL_QUEUE}/{_MINIMAX_MODEL}/requests/{request_ids[i]}/status",
                    headers=headers, timeout=15,
                ).json()
                status = st.get("status", "")
                if status == "COMPLETED":
                    result = requests.get(
                        f"{_FAL_QUEUE}/{_MINIMAX_MODEL}/requests/{request_ids[i]}",
                        headers=headers, timeout=15,
                    ).json()
                    video_url = result["video"]["url"]
                    clip_path = tmp / f"c{i:02d}.mp4"
                    clip_path.write_bytes(requests.get(video_url, timeout=120).content)
                    clips[i] = clip_path
                    pending.discard(i)
                    print(f"   ✓ Clip {i+1}/{len(request_ids)}")
                elif status == "FAILED":
                    raise RuntimeError(f"MiniMax clip {i+1} failed: {st.get('error', '')}")
            except RuntimeError:
                raise
            except Exception:
                pass  # transient network error — retry next poll cycle

    return [c for c in clips if c is not None]  # type: ignore[misc]


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
    """Ken Burns zoompan fallback — used when AI animation is unavailable."""
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
    """Concat AI/Ken-Burns clips + mux audio + burn karaoke captions → final MP4."""
    print("[5/5] Assembling final video...")
    tmp     = clip_paths[0].parent  # clips are already in tmp dir
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

    # Clean up tmp dir, images, audio
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
    """Ken Burns fallback path — used when AI animation is skipped."""
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

        # Make viewable by anyone with the link
        service.permissions().create(fileId=f["id"], body={"type": "anyone", "role": "reader"}).execute()

        url = f.get("webViewLink")
        print(f"   ☁️  Drive: {url}")
        return url
    except Exception as e:
        print(f"   Drive upload failed (non-fatal): {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_video(topic: str) -> tuple[Path, str]:
    slug              = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"
    script            = generate_script(topic)
    audio, wds        = generate_tts(script["script"], slug)
    image_paths, urls = generate_images(script["visual_prompts"], slug)
    tmp               = OUTPUT_DIR / f"{slug}_tmp"
    tmp.mkdir(exist_ok=True)
    clip_paths        = animate_images_ai(urls, slug)
    video             = assemble_clips(clip_paths, audio, wds, slug, cleanup_images=image_paths)
    caption           = script.get("tiktok_caption", "")
    print(f"\n📋 Caption: {caption}")
    return video, caption


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "honey never expires — found in 3000-year-old Egyptian tombs"
    generate_video(topic)
