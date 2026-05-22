"""
@provenweird TikTok Video Pipeline
Input:  topic string  e.g. "honey never expires"
Output: MP4 file in ./output/
"""

import os, json, asyncio, textwrap, math, requests, io
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

load_dotenv()

# Use bundled ffmpeg binary (no system ffmpeg needed)
import imageio_ffmpeg
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_W, VIDEO_H = 1080, 1920   # 9:16 portrait
FPS = 30

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

    # Extract base64 image from response
    for part in data["candidates"][0]["content"]["parts"]:
        if part.get("inlineData"):
            import base64
            img_bytes = base64.b64decode(part["inlineData"]["data"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Resize/crop to 9:16
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


# ─── Step 4: Video assembly ───────────────────────────────────────────────────

def build_caption_frames(script: str, total_frames: int) -> list[str]:
    words = script.split()
    chunk_size = 5
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    frames_per_chunk = total_frames / len(chunks)
    frame_captions = []
    for i in range(total_frames):
        idx = min(int(i / frames_per_chunk), len(chunks) - 1)
        frame_captions.append(chunks[idx])
    return frame_captions


def draw_text_on_frame(
    base_img: np.ndarray,
    text: str,
    hook: str = "",
    show_hook: bool = False,
) -> np.ndarray:
    img = Image.fromarray(base_img)
    draw = ImageDraw.Draw(img)

    # Try system fonts
    def get_font(size):
        for name in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]:
            if os.path.exists(name):
                return ImageFont.truetype(name, size)
        return ImageFont.load_default()

    # Dark gradient overlay at bottom for readability
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    grad_top = VIDEO_H - 500
    for y in range(grad_top, VIDEO_H):
        alpha = int(180 * (y - grad_top) / (VIDEO_H - grad_top))
        ov_draw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Caption text at bottom
    font_cap = get_font(62)
    wrapped = textwrap.fill(text, width=22)
    lines = wrapped.split("\n")
    line_h = 75
    total_h = len(lines) * line_h
    y_start = VIDEO_H - total_h - 90

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_cap)
        tw = bbox[2] - bbox[0]
        x = (VIDEO_W - tw) // 2
        # Black outline
        for dx, dy in [(-3,0),(3,0),(0,-3),(0,3),(-2,-2),(2,-2),(-2,2),(2,2)]:
            draw.text((x+dx, y_start+dy), line, font=font_cap, fill=(0, 0, 0))
        draw.text((x, y_start), line, font=font_cap, fill=(255, 255, 255))
        y_start += line_h

    # Hook text at top (first 3 seconds)
    if show_hook and hook:
        font_hook = get_font(52)
        wrapped_hook = textwrap.fill(hook, width=24)
        hlines = wrapped_hook.split("\n")
        y_h = 180
        for hl in hlines:
            bbox = draw.textbbox((0, 0), hl, font=font_hook)
            tw = bbox[2] - bbox[0]
            x = (VIDEO_W - tw) // 2
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                draw.text((x+dx, y_h+dy), hl, font=font_hook, fill=(0, 0, 0))
            draw.text((x, y_h), hl, font=font_hook, fill=(255, 230, 50))
            y_h += 65

    return np.array(img)


def assemble_video(image_path: Path, audio_path: Path, script_data: dict, slug: str) -> Path:
    print("[4/4] Assembling video...")
    from moviepy import VideoClip, AudioFileClip, ImageClip

    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    total_frames = int(duration * FPS)

    base = np.array(Image.open(image_path).convert("RGB"))
    captions = build_caption_frames(script_data["script"], total_frames)
    hook = script_data.get("hook", "")

    # Ken Burns: gentle zoom from 1.0 → 1.08
    def make_frame(t):
        frame_idx = min(int(t * FPS), total_frames - 1)
        progress = t / duration
        zoom = 1.0 + 0.08 * progress

        # Crop centre based on zoom
        h, w = base.shape[:2]
        new_w = int(w / zoom)
        new_h = int(h / zoom)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = base[y1:y1+new_h, x1:x1+new_w]
        zoomed = np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))

        caption = captions[frame_idx]
        show_hook = t < 3.0
        return draw_text_on_frame(zoomed, caption, hook, show_hook)

    video = VideoClip(make_frame, duration=duration).with_fps(FPS)
    video = video.with_audio(audio)

    out_path = OUTPUT_DIR / f"{slug}.mp4"
    video.write_videofile(
        str(out_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    audio.close()
    print(f"\n✅ Video saved: {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_video(topic: str) -> Path:
    import re, time
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
