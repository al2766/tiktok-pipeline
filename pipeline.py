"""
@provenweird TikTok Video Pipeline
Input:  topic string  e.g. "honey never expires"
Output: MP4 file in ./output/
"""

import os, json, asyncio, textwrap, requests, io, subprocess, re, gc
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

VIDEO_W, VIDEO_H = 1080, 1920
FPS = 24


# ─── Font loading ─────────────────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
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
        "Return ONLY valid JSON (no markdown, no backticks) with these exact keys:\n"
        "  hook: opening line, max 12 words, counterintuitive and surprising\n"
        "  script: full narration, 120-160 words, punchy sentences, no filler\n"
        "  visual_prompts: array of exactly 3 strings — one image prompt per scene. "
        "    Each must be a vivid CARTOON ILLUSTRATION style prompt: bold outlines, "
        "    bright saturated colours, dynamic action pose, cinematic lighting. "
        "    Scene 1 = setup/subject intro. Scene 2 = the key action/fact. Scene 3 = the mind-blowing conclusion. "
        "    NO people, NO text in images.\n"
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
                    "max_tokens": 800,
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
            data = json.loads(raw.strip())
            # Backwards-compat: if single visual_prompt returned, wrap it
            if "visual_prompt" in data and "visual_prompts" not in data:
                data["visual_prompts"] = [data["visual_prompt"]] * 3
            return data
    raise RuntimeError("All Claude models overloaded")


# ─── Step 2: Images (3 scenes) ───────────────────────────────────────────────

def _generate_single_image(prompt: str, path: Path) -> Path:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3-pro-image-preview:generateContent?key={GEMINI_API_KEY}"
    )
    resp = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["image", "text"]},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    for part in data["candidates"][0]["content"]["parts"]:
        if part.get("inlineData"):
            import base64
            img_bytes = base64.b64decode(part["inlineData"]["data"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = _crop_to_ratio(img, VIDEO_W, VIDEO_H)
            img.save(path, quality=95)
            return path
    raise RuntimeError("No image returned from Imagen 3")


def generate_images(visual_prompts: list[str], slug: str) -> list[Path]:
    print(f"[2/4] Generating {len(visual_prompts)} scene images via Imagen 3...")
    paths = []
    for i, prompt in enumerate(visual_prompts):
        path = OUTPUT_DIR / f"{slug}_scene{i}.jpg"
        print(f"   Scene {i+1}/{len(visual_prompts)}…")
        _generate_single_image(prompt, path)
        paths.append(path)
    return paths


def _crop_to_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    target_ratio = target_w / target_h
    w, h = img.size
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        img = img.crop(((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        img = img.crop((0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h))
    return img.resize((target_w, target_h), Image.LANCZOS)


# ─── Step 3: TTS with word-boundary timing ───────────────────────────────────

async def _tts_stream(text: str, path: Path) -> list[dict]:
    """Returns audio file + list of {word, start, end} with real timestamps."""
    import edge_tts
    communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
    boundaries = []
    audio = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"] / 1e7        # 100ns → seconds
            dur   = chunk["duration"] / 1e7
            boundaries.append({
                "word":  chunk["text"],
                "start": start,
                "end":   start + dur,
            })

    path.write_bytes(bytes(audio))
    return boundaries


def generate_tts(script: str, slug: str) -> tuple[Path, list[dict]]:
    print("[3/4] Generating TTS with word timing (en-GB-RyanNeural)...")
    path = OUTPUT_DIR / f"{slug}_audio.mp3"
    boundaries = asyncio.run(_tts_stream(script, path))
    print(f"   Got {len(boundaries)} word boundaries")
    return path, boundaries


def _build_timed_chunks(boundaries: list[dict], chunk_size: int = 4) -> list[dict]:
    """Group word boundaries into caption chunks with exact start/end times."""
    chunks = []
    for i in range(0, len(boundaries), chunk_size):
        group = boundaries[i : i + chunk_size]
        chunks.append({
            "text":  " ".join(w["word"] for w in group),
            "start": group[0]["start"],
            "end":   group[-1]["end"],
        })
    return chunks


def _caption_at(t: float, chunks: list[dict]) -> str:
    for c in chunks:
        if c["start"] <= t < c["end"]:
            return c["text"]
    if chunks and t >= chunks[-1]["end"]:
        return chunks[-1]["text"]
    return ""


# ─── Step 4: Video assembly ───────────────────────────────────────────────────

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
    pil = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grad_top = VIDEO_H - 500
    for y in range(grad_top, VIDEO_H):
        alpha = int(180 * (y - grad_top) / (VIDEO_H - grad_top))
        draw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
    return np.array(Image.alpha_composite(pil, overlay).convert("RGB"))


def _crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Blend two frames. t=0 → all a, t=1 → all b."""
    return (a * (1 - t) + b * t).astype(np.uint8)


def assemble_video(
    image_paths: list[Path],
    audio_path: Path,
    word_boundaries: list[dict],
    script_data: dict,
    slug: str,
) -> Path:
    print("[4/4] Assembling video...")

    duration    = _get_audio_duration(audio_path)
    total_frames = int(duration * FPS)
    FADE_FRAMES  = int(0.4 * FPS)   # 0.4s crossfade between scenes

    # Pre-bake gradient into each scene image
    scenes = [_bake_gradient(np.array(Image.open(p).convert("RGB"))) for p in image_paths]
    gc.collect()

    # Scene timing: split duration evenly across scenes
    n_scenes = len(scenes)
    scene_duration = duration / n_scenes

    # Caption chunks from real word timing
    timed_chunks = _build_timed_chunks(word_boundaries) if word_boundaries else []

    hook       = script_data.get("hook", "")
    font_cap   = get_font(64)
    font_hook  = get_font(54)

    def make_frame(frame_idx: int) -> np.ndarray:
        t        = frame_idx / FPS
        progress = t / duration

        # Which scene + Ken Burns zoom per scene
        scene_idx    = min(int(t / scene_duration), n_scenes - 1)
        scene_t      = (t - scene_idx * scene_duration) / scene_duration  # 0→1 within scene
        zoom         = 1.0 + 0.06 * scene_t
        base         = scenes[scene_idx]

        h, w = base.shape[:2]
        new_w = int(w / zoom)
        new_h = int(h / zoom)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        zoomed = np.array(Image.fromarray(base[y1:y1+new_h, x1:x1+new_w]).resize((w, h), Image.BILINEAR))

        # Crossfade into next scene near scene boundary
        frames_into_scene = frame_idx - int(scene_idx * scene_duration * FPS)
        frames_in_scene   = int(scene_duration * FPS)
        if scene_idx < n_scenes - 1 and frames_into_scene >= frames_in_scene - FADE_FRAMES:
            fade_t   = (frames_into_scene - (frames_in_scene - FADE_FRAMES)) / FADE_FRAMES
            next_base = scenes[scene_idx + 1]
            next_zoomed = np.array(Image.fromarray(next_base).resize((w, h), Image.BILINEAR))
            zoomed = _crossfade(zoomed, next_zoomed, min(fade_t, 1.0))

        # Draw captions
        img  = Image.fromarray(zoomed)
        draw = ImageDraw.Draw(img)

        caption = _caption_at(t, timed_chunks)
        if caption:
            wrapped = textwrap.fill(caption, width=20)
            lines   = wrapped.split("\n")
            line_h  = 80
            y_start = VIDEO_H - len(lines) * line_h - 100
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font_cap)
                tw   = bbox[2] - bbox[0]
                x    = (VIDEO_W - tw) // 2
                for dx, dy in [(-3,0),(3,0),(0,-3),(0,3),(-2,-2),(2,-2),(-2,2),(2,2)]:
                    draw.text((x+dx, y_start+dy), line, font=font_cap, fill=(0, 0, 0))
                draw.text((x, y_start), line, font=font_cap, fill=(255, 255, 255))
                y_start += line_h

        # Hook at top for first 3s
        if t < 3.0 and hook:
            wrapped_hook = textwrap.fill(hook, width=22)
            y_h = 180
            for hl in wrapped_hook.split("\n"):
                bbox = draw.textbbox((0, 0), hl, font=font_hook)
                tw   = bbox[2] - bbox[0]
                x    = (VIDEO_W - tw) // 2
                for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                    draw.text((x+dx, y_h+dy), hl, font=font_hook, fill=(0, 0, 0))
                draw.text((x, y_h), hl, font=font_hook, fill=(255, 230, 50))
                y_h += 65

        return np.array(img)

    # Write silent video via imageio → FFmpeg pipe
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

    # Mux audio (copy video stream — no re-encode)
    subprocess.run(
        [FFMPEG_EXE, "-y",
         "-i", str(silent_path),
         "-i", str(audio_path),
         "-c:v", "copy", "-c:a", "aac", "-shortest",
         str(out_path)],
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

    script_data  = generate_script(topic)
    print(f"   Hook:   {script_data['hook']}")
    print(f"   Scenes: {len(script_data.get('visual_prompts', []))}")

    image_paths = generate_images(script_data["visual_prompts"], slug)
    audio_path, word_boundaries = generate_tts(script_data["script"], slug)
    video_path  = assemble_video(image_paths, audio_path, word_boundaries, script_data, slug)

    print(f"\n📋 TikTok caption:\n{script_data['tiktok_caption']}")
    return video_path


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "honey never expires and archaeologists found edible honey in Egyptian tombs"
    generate_video(topic)
