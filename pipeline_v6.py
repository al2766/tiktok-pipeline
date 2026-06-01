"""
@provenweird TikTok Video Pipeline v6 — Continuous animated story

Changes from v5:
- 7 keyframes instead of 12 sentences-worth of images
- MiniMax fl2v (first+last frame) animates between each consecutive pair
  frame[0]→frame[1], frame[1]→frame[2] ... frame[5]→frame[6]
  = 6 clips × 10 seconds = 60 seconds, zero visible cuts
- No Ken Burns anywhere
- Script prompt updated: keyframes must share same subject/setting/light
  with small positional changes between adjacent frames so MiniMax can
  interpolate smooth motion rather than hard-jump between scenes
"""

import os, re, time, base64, requests, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pipeline import (
    generate_tts,
    generate_images,
    assemble_clips,
    OUTPUT_DIR,
)
from pipeline_v5 import upload_to_drive   # OAuth Drive upload

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE    = "https://api.minimax.io/v1"
FL2V_MODEL      = "MiniMax-Hailuo-02"
FL2V_DURATION   = 6       # 6 s per clip (10 s requires higher plan)
N_KEYFRAMES     = 11      # 11 frames → 10 clips → 60 s

# Gentle camera moves for each consecutive pair — monotone keeps motion smooth
_TRANSITION_PROMPTS = [
    "slow cinematic zoom in, smooth continuous drift toward subject",
    "gradual pull back revealing more of the scene, steady motion",
    "slow push forward into the scene, soft depth of field",
    "gentle camera float upward, smooth atmospheric drift",
    "slow pan across the subject, steady cinematic movement",
    "smooth zoom out to wide shot, continuous flowing motion",
    "slow cinematic zoom in, gentle drift",
    "gradual pull back, smooth steady reveal",
    "slow dolly forward, soft depth of field",
    "gentle camera float, smooth atmospheric motion",
]

# ─── Script system (v6) ───────────────────────────────────────────────────────

_SCRIPT_SYSTEM = """You write scripts for @provenweird — a psychology and science TikTok focused on facts that make people say "wait, really?" and feel genuinely better informed.

--- VOICE ---

Natural and genuinely curious. Like a friend who cannot stop thinking about what they just found out. The facts do the work. No wordplay, no forced analogies, no punchlines. State things plainly and trust they land. Warm but matter-of-fact.

--- WRITING FOR AI VOICE — FOLLOW EXACTLY ---

This script is read by an AI text-to-speech system. These rules exist because of how AI voices process text:

1. Short sentences. One idea per sentence. Maximum 12 words per sentence.
2. Never use ellipsis (...) as a dramatic pause. AI voices do not pause for them — it sounds broken.
3. Never use em-dashes (—) as stylistic pauses. Same problem.
4. Pacing comes from sentence length and paragraph breaks, not punctuation tricks.
5. A short standalone sentence followed by a new paragraph naturally creates a pause. Use that instead.
6. Commas create a brief breath. Periods create a full stop. Paragraph break creates a natural pause.
7. If something is important, make it its own short sentence. Do not decorate it with punctuation.

--- SCRIPT STRUCTURE — 12 sentences, 100 to 130 words ---

Sentences 1 to 2: Hook. Genuine surprise or curiosity. Do not reveal the full fact yet.
Sentences 3 to 4: The familiar version. What they think they know.
Sentences 5 to 6: The actual fact, stated plainly. Let it land on its own.
Sentence 7: Exactly one specific number or measurement. The credibility anchor.
Sentences 8 to 10: The mechanism. How and why, in plain language.
Sentences 11 to 12: The takeaway. How they will now see this differently. Not a question. Not a punchline.

RULES: 100-130 words total. Exactly one number. No forced humour. No question at the end. Halal content only.

---

KEYFRAMES — 7 IMAGES FOR ONE CONTINUOUS ANIMATION

These 7 images will be fed to MiniMax AI which animates the motion BETWEEN each consecutive pair:
  Frame 1 → Frame 2: MiniMax generates 10 seconds of smooth animation
  Frame 2 → Frame 3: MiniMax generates 10 seconds of smooth animation
  Frame 3 → Frame 4 ... and so on through Frame 6 → Frame 7

Because Frame 2 ends clip 1 AND starts clip 2, there are zero cuts in the final video. One continuous 60-second animated story.

CRITICAL RULES — READ CAREFULLY:

Rule 1: All 7 frames MUST show the SAME subject in the SAME setting with the SAME lighting and color palette. They are moments of one continuous animation, not 7 different scenes.

Rule 2: The change between adjacent frames must be SMALL. A slight camera angle change, a zoom level, a moment in time moving forward. Large jumps between frames create broken-looking animation.

Rule 3: Think of it exactly like key poses in traditional animation. MiniMax fills in all the motion between each pair. Give it frames it can smoothly interpolate.

Rule 4: The frames should roughly follow the script's narrative arc — early frames establish, middle frames reveal the key fact visually, later frames show consequence or scale.

Rule 5: Accuracy to what is being described matters — if the script talks about a brain, show a brain. But keep the same visual world across all 7 frames.

EXAMPLE KEYFRAME ARC for a cat meowing topic (11 frames):
  Frame 1: wide shot, warm living room, cat sitting on sofa facing camera, soft natural light
  Frame 2: slightly closer, same room, same cat, ears forward attentively
  Frame 3: medium shot of cat's face, same warm light, eyes wide
  Frame 4: slightly tighter, cat's face, mouth just opening
  Frame 5: close-up of cat's mouth and whiskers, same lighting
  Frame 6: close-up pulling back, cat and human hand entering frame
  Frame 7: medium shot, cat looking up at human, warm natural light
  Frame 8: medium shot, human leaning down slightly toward cat
  Frame 9: slightly wider, both cat and human in frame, warm light
  Frame 10: medium shot, cat settled, relaxed posture, same room
  Frame 11: wide shot, same as frame 1 composition, cat curled up, human nearby, calm

Notice: same room, same cat, same lighting across all 7. Changes are small and progressive.

Image format for all 7: photorealistic, 9:16 vertical, cinematic, consistent warm natural lighting, no text in image.

Return ONLY valid JSON, no markdown, no backticks:
{"hook":"first sentence only","sentences":["exactly 12 strings"],"visual_prompts":["exactly 11 keyframe image prompts — same subject/setting/lighting across all 11, small changes between adjacent frames, photorealistic 9:16 vertical no text"],"tiktok_caption":"punchy 1-line caption + 5 hashtags"}"""


# ─── Script generation ────────────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    print(f"[1/5] Generating script + keyframes: {topic}")
    for attempt in range(3):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":        os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "anthropic-beta":   "prompt-caching-2024-07-31",
                "content-type":     "application/json",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 2000,
                "system": [{"type": "text", "text": _SCRIPT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": f"Topic: {topic}"}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
            data["script"] = " ".join(data["sentences"])
            print(f"   Hook: {data['hook']}")
            print(f"   Keyframes: {len(data.get('visual_prompts', []))}")
            return data
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError("Script generation failed after 3 attempts")


# ─── MiniMax fl2v helpers ─────────────────────────────────────────────────────

def _fl2v_submit(first_path: Path, last_path: Path, prompt: str) -> str:
    first_b64 = base64.b64encode(first_path.read_bytes()).decode()
    last_b64  = base64.b64encode(last_path.read_bytes()).decode()
    # Retry up to 5 times on RPM rate limit (status 1002)
    for attempt, wait in enumerate([0, 15, 30, 60, 120]):
        if wait:
            print(f"   Rate limit hit — waiting {wait}s before retry {attempt}…")
            time.sleep(wait)
        resp = requests.post(
            f"{MINIMAX_BASE}/video_generation",
            headers={"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"},
            json={
                "model":             FL2V_MODEL,
                "first_frame_image": f"data:image/jpeg;base64,{first_b64}",
                "last_frame_image":  f"data:image/jpeg;base64,{last_b64}",
                "prompt":            prompt,
                "prompt_optimizer":  True,
                "duration":          FL2V_DURATION,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data        = resp.json()
        status_code = data.get("base_resp", {}).get("status_code", -1)
        if status_code == 0:
            return str(data["task_id"])
        if status_code == 1002:
            continue   # rate limit — retry after wait
        raise RuntimeError(
            f"MiniMax fl2v submit failed (status {status_code}): "
            f"{data.get('base_resp', {}).get('status_msg')}"
        )
    raise RuntimeError("MiniMax fl2v rate limit persisted after 5 retries")


def _fl2v_poll(task_id: str, timeout: int = 500) -> str:
    """Poll until done. Returns video download URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(12)
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
        print(f"   [{task_id}] {status}...")
    raise RuntimeError(f"MiniMax task {task_id} timed out after {timeout}s")


# ─── Animation: one continuous fl2v story ────────────────────────────────────

def animate_images(image_paths: list[Path], slug: str) -> list[Path]:
    """
    7 keyframes → 6 fl2v clips of 10 s each = 60 s continuous story.
    Each clip: frame[i] → frame[i+1], no cuts between clips.
    """
    n_clips = len(image_paths) - 1
    print(f"[4/5] Generating {n_clips} continuous fl2v clips ({FL2V_DURATION}s each)...")

    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not set — add it to .env")

    clip_dir = OUTPUT_DIR / f"{slug}_clips"
    clip_dir.mkdir(exist_ok=True)

    # Submit all tasks upfront so they queue in parallel
    task_ids: list[str] = []
    for i in range(n_clips):
        prompt  = _TRANSITION_PROMPTS[i % len(_TRANSITION_PROMPTS)]
        print(f"   Submitting clip {i + 1}/{n_clips} (frame {i + 1}→{i + 2})...")
        task_id = _fl2v_submit(image_paths[i], image_paths[i + 1], prompt)
        task_ids.append(task_id)
        time.sleep(5)   # 5s gap between submissions to respect RPM limit

    # Poll and download in submission order
    clip_paths: list[Path] = []
    for i, task_id in enumerate(task_ids):
        print(f"   Waiting for clip {i + 1}/{n_clips}...")
        video_url  = _fl2v_poll(task_id)
        clip_path  = clip_dir / f"clip_{i:02d}.mp4"
        r          = requests.get(video_url, timeout=180)
        r.raise_for_status()
        clip_path.write_bytes(r.content)
        print(f"   ✓ Clip {i + 1} saved ({len(r.content) // 1024} KB)")
        clip_paths.append(clip_path)

    return clip_paths


# ─── Entry point ──────────────────────────────────────────────────────────────

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
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "why your cat only meows at humans not other cats"
    generate_video(topic)
