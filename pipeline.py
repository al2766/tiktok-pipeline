"""
@provenweird TikTok Video Pipeline
Input:  topic string  e.g. "honey never expires"
Output: MP4 file in ./output/
"""

import os, json, asyncio, textwrap, math, requests, io, subprocess, re, gc
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio
import imageio_ffmpeg

load_dotenv()

os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_W, VIDEO_H = 1080, 1920   # 9:16 portrait
FPS = 24  # dropped from 30 — fewer frames, less memory pressure


# ─── Font loading ─────────────────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        # Linux (Render)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── Step 1: Script ──────────────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    import time
    print(f"[1/4] Generating script for: {topic}")
    system = (
        "You write TikTok scripts for @provenweird — a science facts channel. "
        "Return ONLY valid JSON (no markdown, no backticks) with these keys:\n"
        "  hook: opening line, max 12 words, counterintuitive and surprising\n"
        "  script: full narration, 120-160 words, punchy sentences, no filler\n"
        "  visual_prompt: image prompt for AI (abstract/scientific/surreal scene, NO people, NO text, vivid colours)\n"
        "  tiktok_caption: post caption with 5 relevant hashtags\n"
        "Halal: no haram topics. No music references. Facts only."
    )
    for model in ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]:
        for attempt in range(3):
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 600,
                    "system": system,
                    "messages": [{"role": "user", "content": f"Topic: {topic}"}],
                },
                timeout=30,
            )
            if resp.status_code == 529:
                print(f"   {model} overloaded, retrying...")
                time.sleep(5)
                continue
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
    raise RuntimeError("All Claude models overloaded")


# ─── Step 2: Image (Imagen 3 via Gemini API) ─────────────────────────────────

def generate_image(visual_prompt: str, slug: str) -> Path:
    print("[2/4] Generating background image via Imagen 3...")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3-pro-image-preview:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": visual_prompt}]}],
        "generationConfig": {"responseModalities": ["image", "text"]},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    for part in data["candidates"][0]["content"]["parts"]:
        if part.get("inlineData"):
            import base64
            img_bytes = base64.b64decode(part["inlineData"]["data"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = crop_to_ratio(img, VIDEO_W, VIDEO_H)
            path = OUTPUT_DIR / f"{slug}_bg.jpg"
            img.save(path, quality=95)
            return path

    raise RuntimeError("No image returned from Imagen 3")


def crop_to_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    target_ratio = target_w / target_h
    w, h = img.size
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        img = img.crop((x, 0, x + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        img = img.crop((0, y, w, y + new_h))
    return img.resize((target_w, target_h), Image.LANCZOS)


# ─── Step 3: TTS (MS Edge TTS — free) ────────────────────────────────────────

async def _tts(text: str, path: Path):
    import edge_tts
    communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
    await communicate.save(str(path))

def generate_tts(script: str, slug: str) -> Path:
    print("[3/4] Generating TTS audio (en-GB-RyanNeural)...")
    path = OUTPUT_DIR / f"{slug}_audio.mp3"
    asyncio.run(_tts(script, path))
    return path


# ─── Step 4: Video assembly (imageio + FFmpeg — no moviepy) ──────────────────

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


def _bake_gradient(img: np.ndarray) -> np.ndarray:
    """Apply bottom gradient overlay once — avoids per-frame RGBA compositing."""
    pil = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grad_top = VIDEO_H - 500
    for y in range(grad_top, VIDEO_H):
        alpha = int(180 * (y - grad_top) / (VIDEO_H - grad_top))
        draw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
    result = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.array(result)


def assemble_video(image_path: Path, audio_path: Path, script_data: dict, slug: str) -> Path:
    print("[4/4] Assembling video...")

    duration = _get_audio_duration(audio_path)
    total_frames = int(duration * FPS)

    # Pre-bake gradient into base image once (eliminates per-frame RGBA overhead)
    raw_base = np.array(Image.open(image_path).convert("RGB"))
    base = _bake_gradient(raw_base)
    del raw_base
    gc.collect()

    # Pre-load fonts once
    font_cap  = get_font(62)
    font_hook = get_font(52)

    # Build caption list once
    words = script_data["script"].split()
    chunk_size = 5
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    frames_per_chunk = total_frames / len(chunks)
    hook = script_data.get("hook", "")

    def make_frame(frame_idx: int) -> np.ndarray:
        t = frame_idx / FPS
        progress = t / duration
        zoom = 1.0 + 0.08 * progress

        # Ken Burns crop
        h, w = base.shape[:2]
        new_w = int(w / zoom)
        new_h = int(h / zoom)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = base[y1:y1+new_h, x1:x1+new_w]
        zoomed = np.array(Image.fromarray(cropped).resize((w, h), Image.BILINEAR))

        # Draw text directly (gradient already baked — no RGBA needed)
        img = Image.fromarray(zoomed)
        draw = ImageDraw.Draw(img)

        chunk_idx = min(int(frame_idx / frames_per_chunk), len(chunks) - 1)
        caption = chunks[chunk_idx]
        wrapped = textwrap.fill(caption, width=22)
        lines = wrapped.split("\n")
        line_h = 75
        y_start = VIDEO_H - len(lines) * line_h - 90

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_cap)
            tw = bbox[2] - bbox[0]
            x = (VIDEO_W - tw) // 2
            for dx, dy in [(-3,0),(3,0),(0,-3),(0,3),(-2,-2),(2,-2),(-2,2),(2,2)]:
                draw.text((x+dx, y_start+dy), line, font=font_cap, fill=(0, 0, 0))
            draw.text((x, y_start), line, font=font_cap, fill=(255, 255, 255))
            y_start += line_h

        if t < 3.0 and hook:
            wrapped_hook = textwrap.fill(hook, width=24)
            y_h = 180
            for hl in wrapped_hook.split("\n"):
                bbox = draw.textbbox((0, 0), hl, font=font_hook)
                tw = bbox[2] - bbox[0]
                x = (VIDEO_W - tw) // 2
                for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                    draw.text((x+dx, y_h+dy), hl, font=font_hook, fill=(0, 0, 0))
                draw.text((x, y_h), hl, font=font_hook, fill=(255, 230, 50))
                y_h += 65

        return np.array(img)

    # Write video frames via imageio (streams directly to FFmpeg — no moviepy needed)
    silent_path = OUTPUT_DIR / f"{slug}_silent.mp4"
    out_path    = OUTPUT_DIR / f"{slug}.mp4"

    writer = imageio.get_writer(
        str(silent_path),
        fps=FPS,
        codec="libx264",
        quality=None,
        ffmpeg_params=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
    )
    try:
        for i in range(total_frames):
            writer.append_data(make_frame(i))
    finally:
        writer.close()

    # Mux audio in a second pass (copy video stream — no re-encode)
    subprocess.run(
        [
            FFMPEG_EXE, "-y",
            "-i", str(silent_path),
            "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    silent_path.unlink(missing_ok=True)

    print(f"\n✅ Video saved: {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_video(topic: str) -> Path:
    import time
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"

    script_data = generate_script(topic)
    print(f"   Hook:   {script_data['hook']}")
    print(f"   Words:  {len(script_data['script'].split())}")

    image_path = generate_image(script_data["visual_prompt"], slug)
    audio_path = generate_tts(script_data["script"], slug)
    video_path = assemble_video(image_path, audio_path, script_data, slug)

    print(f"\n📋 TikTok caption:\n{script_data['tiktok_caption']}")
    return video_path


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "honey never expires and archaeologists found edible honey in Egyptian tombs"
    generate_video(topic)
