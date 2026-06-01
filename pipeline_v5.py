"""
@provenweird TikTok Video Pipeline v5

Changes from v4:
- Rewritten _SCRIPT_SYSTEM in pipeline.py:
  - AI TTS rules: short sentences, no ellipsis/dash tricks, pacing via structure
  - Voice: natural curiosity, no forced humour or wordplay
  - Topic focus: psychology, human behavior, everyday phenomena
  - Visual prompts: accuracy to sentence is priority, continuity is secondary
- Animation: same MiniMax x4 + Ken Burns x8 mix
"""

import os, time, base64, requests, subprocess, re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Re-export unchanged pipeline steps so server.py can import everything from here
from pipeline import (
    generate_script,
    generate_tts,
    generate_images,
    assemble_clips,
    OUTPUT_DIR,
    VIDEO_W,
    VIDEO_H,
    FFMPEG,
    _ZOOMPAN,
)

GDRIVE_FOLDER_ID = "1TcTICXwpiN7st-HPXYaTlDEgfJcFdw4y"
_TOKEN_PATH      = Path.home() / ".gdrive_token.json"


def upload_to_drive(video_path: Path) -> str | None:
    """OAuth-based Drive upload. Uses ~/.gdrive_token.json."""
    if not _TOKEN_PATH.exists():
        print("   No Drive token — skipping upload.")
        return None
    try:
        import json as _json
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        token_data = _json.loads(_TOKEN_PATH.read_text())
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes"),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _TOKEN_PATH.write_text(creds.to_json())

        service   = build("drive", "v3", credentials=creds, cache_discovery=False)
        file_meta = {"name": video_path.name, "parents": [GDRIVE_FOLDER_ID]}
        media     = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
        f         = service.files().create(body=file_meta, media_body=media, fields="id,webViewLink").execute()
        service.permissions().create(fileId=f["id"], body={"type": "anyone", "role": "reader"}).execute()
        url = f.get("webViewLink")
        print(f"   Drive: {url}")
        return url
    except Exception as e:
        print(f"   Drive upload failed (non-fatal): {e}")
        return None

# ─── MiniMax config ───────────────────────────────────────────────────────────

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE    = "https://api.minimax.io/v1"
MINIMAX_MODEL   = "MiniMax-Hailuo-2.3-Fast"   # fastest; swap to "MiniMax-Hailuo-2.3" for higher quality

# Scenes to animate with MiniMax. The rest → Ken Burns.
# Every 3rd scene: hook (0), number anchor (3), mechanism peak (6), twist (9)
ANIMATED_INDICES: set[int] = {0, 3, 6, 9}

# Ken Burns clip duration in frames at 25fps.
# 150 frames = 6 s, matching MiniMax's default 6-second output.
_KB_FRAMES = 150

_MOTION_PROMPTS = [
    "slow cinematic zoom in, gentle camera drift",
    "slow pull back reveal, soft ambient motion",
    "subtle camera pan left, dreamy atmosphere",
    "slow zoom out revealing the scene",
    "gentle handheld camera drift, cinematic depth",
    "slow dolly forward, ethereal glow pulsing",
    "camera slowly rotates, atmospheric haze",
    "subtle zoom in, particles floating upward",
    "gentle camera float upward, cinematic light rays",
    "slow pan right, soft bokeh depth of field",
]


# ─── MiniMax helpers ──────────────────────────────────────────────────────────

def _minimax_submit(image_path: Path, prompt: str) -> str:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    resp = requests.post(
        f"{MINIMAX_BASE}/video_generation",
        headers={"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"},
        json={
            "model":               MINIMAX_MODEL,
            "first_frame_image":   f"data:image/jpeg;base64,{img_b64}",
            "prompt":              prompt,
            "prompt_optimizer":    True,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data        = resp.json()
    status_code = data.get("base_resp", {}).get("status_code", -1)
    if status_code != 0:
        raise RuntimeError(f"MiniMax submit failed (status {status_code}): {data.get('base_resp', {}).get('status_msg')}")
    return str(data["task_id"])


def _minimax_poll(task_id: str, timeout: int = 400) -> str:
    """Poll until done. Returns video download URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(10)
        resp = requests.get(
            f"{MINIMAX_BASE}/query/video_generation",
            headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"},
            params={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        data   = resp.json()
        status = data.get("status", "")
        if status == "Success":
            file_id   = data["file_id"]
            file_resp = requests.get(
                f"{MINIMAX_BASE}/files/retrieve",
                headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"},
                params={"file_id": file_id},
                timeout=30,
            )
            file_resp.raise_for_status()
            return file_resp.json()["file"]["download_url"]
        if status == "Fail":
            raise RuntimeError(f"MiniMax task {task_id} failed: {data}")
        print(f"   MiniMax {task_id}: {status}…")
    raise RuntimeError(f"MiniMax task {task_id} timed out after {timeout}s")


# ─── Ken Burns helper ─────────────────────────────────────────────────────────

def _kenburns_clip(img_path: Path, idx: int, tmp: Path) -> Path:
    z    = _ZOOMPAN[idx % 2]
    clip = tmp / f"kb_{idx:02d}.mp4"
    vf   = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"zoompan=z='{z}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
        f":d={_KB_FRAMES}:s={VIDEO_W}x{VIDEO_H}:fps=25"
    )
    r = subprocess.run(
        [FFMPEG, "-y", "-loop", "1", "-i", str(img_path),
         "-vf", vf, "-frames:v", str(_KB_FRAMES),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
         "-tune", "zerolatency", "-threads", "1", "-pix_fmt", "yuv420p",
         str(clip)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Ken Burns clip {idx + 1} failed:\n{r.stderr.decode(errors='replace')[-400:]}"
        )
    return clip


# ─── Main animate function (replaces pipeline.animate_images) ─────────────────

def animate_images(image_paths: list[Path], slug: str) -> list[Path]:
    """
    v4 mixed render:
      - ANIMATED_INDICES → MiniMax Hailuo AI (image-to-video)
      - all others       → Ken Burns zoom/pan (free, local FFmpeg)
    """
    n_mm  = len(ANIMATED_INDICES)
    n_kb  = len(image_paths) - n_mm
    print(f"[4/5] Animating {len(image_paths)} scenes (MiniMax ×{n_mm}, Ken Burns ×{n_kb})…")

    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not set — add it to your .env file (get key from platform.minimax.io)")

    clip_dir = OUTPUT_DIR / f"{slug}_clips"
    clip_dir.mkdir(exist_ok=True)
    tmp = clip_dir / "kb_tmp"
    tmp.mkdir(exist_ok=True)

    # 1. Submit all MiniMax tasks upfront (async — they queue server-side)
    minimax_tasks: dict[int, str] = {}
    for i, img_path in enumerate(image_paths):
        if i in ANIMATED_INDICES:
            motion  = _MOTION_PROMPTS[i % len(_MOTION_PROMPTS)]
            print(f"   Submitting MiniMax scene {i + 1}/{len(image_paths)}…")
            task_id = _minimax_submit(img_path, motion)
            minimax_tasks[i] = task_id
            time.sleep(0.5)

    # 2. Ken Burns for non-animated scenes (fast, local — runs while MiniMax processes)
    kb_paths: dict[int, Path] = {}
    for i, img_path in enumerate(image_paths):
        if i not in ANIMATED_INDICES:
            print(f"   Ken Burns scene {i + 1}/{len(image_paths)}…")
            kb_paths[i] = _kenburns_clip(img_path, i, tmp)

    # 3. Poll + download MiniMax results
    minimax_paths: dict[int, Path] = {}
    for i, task_id in minimax_tasks.items():
        print(f"   Waiting for MiniMax scene {i + 1} (task {task_id})…")
        video_url  = _minimax_poll(task_id)
        clip_path  = clip_dir / f"clip_{i:02d}.mp4"
        r          = requests.get(video_url, timeout=120)
        r.raise_for_status()
        clip_path.write_bytes(r.content)
        print(f"   ✓ Scene {i + 1} saved ({len(r.content) // 1024} KB)")
        minimax_paths[i] = clip_path

    # 4. Return clips in scene order
    result: list[Path] = []
    for i in range(len(image_paths)):
        result.append(minimax_paths[i] if i in minimax_paths else kb_paths[i])
    return result


# ─── Entry point ─────────────────────────────────────────────────────────────

def generate_video(topic: str) -> tuple[Path, str, str | None]:
    slug        = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"
    script      = generate_script(topic)
    audio, wds  = generate_tts(script["script"], slug)
    image_paths = generate_images(script["visual_prompts"], slug)
    clip_paths  = animate_images(image_paths, slug)
    video       = assemble_clips(clip_paths, audio, wds, slug, cleanup_images=image_paths)
    caption     = script.get("tiktok_caption", "")
    drive_url   = upload_to_drive(video)
    print(f"\nCaption:   {caption}")
    print(f"Drive URL: {drive_url or 'not uploaded'}")
    return video, caption, drive_url


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "honey never expires — found in 3000-year-old Egyptian tombs"
    generate_video(topic)
