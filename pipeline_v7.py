"""
@provenweird TikTok Video Pipeline v7

Changes from v5:
- MiniMax replaced with Wan 2.1 via FAL.ai (image-to-video)
- Same FAL key already used for FLUX image generation — zero new setup
- Cost: ~$0.20/clip × 4 clips = ~$0.80/video (vs ~$3+ with Kling or MiniMax issues)
- Quality close to Kling for short-form social content
- Same mixed render: ANIMATED_INDICES get Wan i2v, rest get Ken Burns
"""

import os, re, time, base64, requests, json, subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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
from pipeline_v5 import upload_to_drive

FAL_KEY  = os.environ.get("FAL_KEY", "")
FAL_BASE = "https://queue.fal.run/fal-ai/wan-i2v"

# Every 3rd scene gets Wan i2v (4 of 12). Rest → Ken Burns.
ANIMATED_INDICES: set[int] = {0, 3, 6, 9}

_KB_FRAMES = 150   # 6 s at 25fps, matches Wan's ~5s output

_MOTION_PROMPTS = [
    "slow cinematic zoom in, gentle camera drift, smooth motion",
    "gradual pull back revealing the scene, steady cinematic motion",
    "slow push forward into the scene, soft depth of field",
    "gentle camera float upward, smooth atmospheric movement",
    "slow pan across the subject, steady cinematic movement",
    "subtle zoom in, particles floating, cinematic atmosphere",
    "camera slowly drifts left, dreamy smooth motion",
    "gentle dolly forward, ethereal glow, cinematic",
    "slow zoom out revealing the full scene, smooth motion",
    "gentle camera float, soft bokeh, cinematic depth",
]


# ─── Wan i2v via FAL queue ────────────────────────────────────────────────────

def _wan_submit(image_path: Path, prompt: str) -> str:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    resp = requests.post(
        FAL_BASE,
        headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
        json={
            "prompt":             prompt,
            "image_url":          f"data:image/jpeg;base64,{img_b64}",
            "aspect_ratio":       "9:16",
            "resolution":         "720p",
            "num_frames":         81,
            "frames_per_second":  16,
            "acceleration":       "regular",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["request_id"]


def _wan_poll(request_id: str, timeout: int = 300) -> str:
    """Poll until done. Returns video URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(8)
        status_resp = requests.get(
            f"{FAL_BASE}/requests/{request_id}/status",
            headers={"Authorization": f"Key {FAL_KEY}"},
            timeout=30,
        )
        status_resp.raise_for_status()
        status = status_resp.json().get("status", "")
        if status == "COMPLETED":
            result = requests.get(
                f"{FAL_BASE}/requests/{request_id}",
                headers={"Authorization": f"Key {FAL_KEY}"},
                timeout=30,
            )
            result.raise_for_status()
            return result.json()["video"]["url"]
        if status == "FAILED":
            raise RuntimeError(f"Wan i2v task {request_id} failed")
        print(f"   Wan {request_id[:12]}: {status}…")
    raise RuntimeError(f"Wan i2v {request_id} timed out after {timeout}s")


# ─── Ken Burns fallback ───────────────────────────────────────────────────────

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
        raise RuntimeError(f"Ken Burns clip {idx+1} failed:\n{r.stderr.decode(errors='replace')[-400:]}")
    return clip


# ─── Main animate function ────────────────────────────────────────────────────

def animate_images(image_paths: list[Path], slug: str) -> list[Path]:
    """
    v7: Wan 2.1 i2v via FAL for ANIMATED_INDICES, Ken Burns for the rest.
    """
    n_wan = len(ANIMATED_INDICES)
    n_kb  = len(image_paths) - n_wan
    print(f"[4/5] Animating {len(image_paths)} scenes (Wan i2v ×{n_wan}, Ken Burns ×{n_kb})…")

    if not FAL_KEY:
        raise RuntimeError("FAL_KEY not set — add it to your .env file")

    clip_dir = OUTPUT_DIR / f"{slug}_clips"
    clip_dir.mkdir(exist_ok=True)
    tmp = clip_dir / "kb_tmp"
    tmp.mkdir(exist_ok=True)

    # 1. Submit all Wan tasks upfront (async queue)
    wan_tasks: dict[int, str] = {}
    for i, img_path in enumerate(image_paths):
        if i in ANIMATED_INDICES:
            motion  = _MOTION_PROMPTS[i % len(_MOTION_PROMPTS)]
            print(f"   Submitting Wan scene {i+1}/{len(image_paths)}…")
            request_id = _wan_submit(img_path, motion)
            wan_tasks[i] = request_id
            time.sleep(1)

    # 2. Ken Burns for the rest (fast, local)
    kb_paths: dict[int, Path] = {}
    for i, img_path in enumerate(image_paths):
        if i not in ANIMATED_INDICES:
            print(f"   Ken Burns scene {i+1}/{len(image_paths)}…")
            kb_paths[i] = _kenburns_clip(img_path, i, tmp)

    # 3. Poll + download Wan results
    wan_paths: dict[int, Path] = {}
    for i, request_id in wan_tasks.items():
        print(f"   Waiting for Wan scene {i+1} ({request_id[:12]})…")
        video_url  = _wan_poll(request_id)
        clip_path  = clip_dir / f"clip_{i:02d}.mp4"
        r          = requests.get(video_url, timeout=120)
        r.raise_for_status()
        clip_path.write_bytes(r.content)
        print(f"   ✓ Scene {i+1} saved ({len(r.content) // 1024} KB)")
        wan_paths[i] = clip_path

    # 4. Return in scene order
    result: list[Path] = []
    for i in range(len(image_paths)):
        result.append(wan_paths[i] if i in wan_paths else kb_paths[i])
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
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "why airplane window holes exist"
    generate_video(topic)
