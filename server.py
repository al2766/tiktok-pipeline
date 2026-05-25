"""
Flask API wrapper around the TikTok video pipeline.
POST /generate          { "topic": "..." }               → { "job_id": "..." }
GET  /status/:id                                         → { "status": "...", "video_b64": "...", "caption": "..." }
POST /post              { "job_id": "...", "title": "..." } → { "ok": true, "publish_id": "..." }
GET  /oauth/start                                       → redirect to TikTok OAuth
GET  /oauth/callback                                    → exchange code, store tokens
GET  /oauth/status                                      → { "authorized": true/false }
"""

import os, uuid, threading, base64, json
from pathlib import Path
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from dotenv import load_dotenv
import requests as http_requests

load_dotenv()

# Set bundled ffmpeg before importing pipeline
import imageio_ffmpeg
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

from pipeline import generate_video, generate_script, generate_images, generate_tts, animate_images, assemble_video

app = Flask(__name__)
CORS(app)

jobs: dict = {}   # job_id → { status, video_b64, caption, error }

TIKTOK_CLIENT_KEY    = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
RENDER_URL           = os.environ.get("RENDER_URL", "https://tiktok-pipeline-cgtt.onrender.com")
# sandbox uses video.upload (draft), production uses video.publish (direct post)
TIKTOK_SCOPE         = os.environ.get("TIKTOK_SCOPE", "video.upload,user.info.basic")
TOKEN_FILE           = "/tmp/tiktok_tokens.json"


def _save_tokens(data: dict):
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)


def _load_tokens() -> dict | None:
    try:
        with open(TOKEN_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ─── pipeline job ──────────────────────────────────────────────────────────────

def _run_job(job_id: str, topic: str):
    jobs[job_id]["status"] = "generating"
    try:
        import re, time as _time
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(_time.time())}"

        jobs[job_id]["step"] = "script"
        script_data = generate_script(topic)
        jobs[job_id]["caption"] = script_data.get("tiktok_caption", "")

        jobs[job_id]["step"] = "images"
        image_paths = generate_images(script_data["visual_prompts"], slug)

        jobs[job_id]["step"] = "voice"
        audio_path, word_boundaries = generate_tts(script_data["script"], slug)

        jobs[job_id]["step"] = "animation"
        clip_paths = animate_images(image_paths, slug)

        jobs[job_id]["step"] = "assembly"
        video_path = assemble_video(clip_paths, audio_path, word_boundaries, script_data, slug)

        with open(video_path, "rb") as f:
            jobs[job_id]["video_b64"] = base64.b64encode(f.read()).decode()

        # Cleanup intermediates
        for p in image_paths:
            p.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["step"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")


# ─── generate ──────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic required"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "video_b64": None, "caption": None, "error": None}

    thread = threading.Thread(target=_run_job, args=(job_id, topic), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


# ─── TikTok OAuth ──────────────────────────────────────────────────────────────

@app.route("/oauth/start")
def oauth_start():
    from urllib.parse import urlencode
    params = {
        "client_key": TIKTOK_CLIENT_KEY,
        "response_type": "code",
        "scope": TIKTOK_SCOPE,
        "redirect_uri": f"{RENDER_URL}/oauth/callback",
        "state": "provenweird",
    }
    return redirect("https://www.tiktok.com/v2/auth/authorize/?" + urlencode(params))


@app.route("/oauth/callback")
def oauth_callback():
    code  = request.args.get("code")
    error = request.args.get("error")
    if error or not code:
        return f"<h2>Auth failed: {error or 'no code returned'}</h2>", 400

    resp = http_requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        data={
            "client_key":     TIKTOK_CLIENT_KEY,
            "client_secret":  TIKTOK_CLIENT_SECRET,
            "code":           code,
            "grant_type":     "authorization_code",
            "redirect_uri":   f"{RENDER_URL}/oauth/callback",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    token_data = resp.json()
    if "access_token" not in token_data:
        return f"<h2>Token exchange failed</h2><pre>{token_data}</pre>", 400

    _save_tokens(token_data)
    return "<h2>TikTok authorized ✓</h2><p>You can close this tab. The Post button is now enabled.</p>"


@app.route("/oauth/status")
def oauth_status():
    return jsonify({"authorized": _load_tokens() is not None})


# ─── post to TikTok ────────────────────────────────────────────────────────────

@app.route("/post", methods=["POST"])
def post_to_tiktok():
    data    = request.get_json(force=True)
    job_id  = data.get("job_id", "")
    title   = (data.get("title") or "").strip() or "Mind-blowing science fact 🔬 #science #facts #learnontiktok"

    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Job not found or not complete"}), 400

    tokens = _load_tokens()
    if not tokens:
        return jsonify({"error": "not_authorized", "auth_url": f"{RENDER_URL}/oauth/start"}), 401

    access_token = tokens["access_token"]
    video_bytes  = base64.b64decode(job["video_b64"])
    video_size   = len(video_bytes)

    # Init upload
    init_resp = http_requests.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json; charset=UTF-8",
        },
        json={
            "post_info": {
                "title":                    title[:2200],
                "privacy_level":            "PUBLIC_TO_EVERYONE",
                "disable_duet":             False,
                "disable_comment":          False,
                "disable_stitch":           False,
                "video_cover_timestamp_ms": 1000,
            },
            "source_info": {
                "source":             "FILE_UPLOAD",
                "video_size":         video_size,
                "chunk_size":         video_size,
                "total_chunk_count":  1,
            },
        },
    )
    init_data = init_resp.json()

    if init_data.get("error", {}).get("code", "").lower() != "ok":
        return jsonify({"error": "TikTok init failed", "detail": init_data}), 500

    publish_id = init_data["data"]["publish_id"]
    upload_url = init_data["data"]["upload_url"]

    # Upload video (single chunk)
    up_resp = http_requests.put(
        upload_url,
        data=video_bytes,
        headers={
            "Content-Type":   "video/mp4",
            "Content-Range":  f"bytes 0-{video_size - 1}/{video_size}",
            "Content-Length": str(video_size),
        },
    )

    if up_resp.status_code not in (200, 201, 206):
        return jsonify({"error": "Upload failed", "status": up_resp.status_code}), 500

    return jsonify({"ok": True, "publish_id": publish_id})


# ─── reel analyzer ────────────────────────────────────────────────────────────

analyze_jobs: dict = {}   # job_id → { status, result, error }

def _run_analyze(job_id: str, url: str, frames: bool):
    analyze_jobs[job_id]["status"] = "analyzing"
    try:
        from reel_analyzer import analyze_reel
        result = analyze_reel(url, extract_frames=frames)
        analyze_jobs[job_id]["status"] = "done"
        analyze_jobs[job_id]["result"] = result
    except Exception as e:
        analyze_jobs[job_id]["status"] = "error"
        analyze_jobs[job_id]["error"] = str(e)
        print(f"Analyze job {job_id} failed: {e}")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST { "url": "https://...", "frames": true }
    Returns job_id immediately; poll /analyze/status/:id for result.
    """
    data   = request.get_json(force=True)
    url    = (data.get("url") or "").strip()
    frames = data.get("frames", True)
    if not url:
        return jsonify({"error": "url required"}), 400

    job_id = str(uuid.uuid4())
    analyze_jobs[job_id] = {"status": "pending", "result": None, "error": None}
    threading.Thread(target=_run_analyze, args=(job_id, url, frames), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/analyze/status/<job_id>")
def analyze_status(job_id: str):
    job = analyze_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


# ─── topic suggestions ────────────────────────────────────────────────────────

# Channels to visit via Playwright — 3 per call, positions 30/31/32 = 9 topics
INSPIRATION_ACCOUNTS = [
    "https://www.tiktok.com/@scitrendz",
    "https://www.tiktok.com/@factsverse",
    "https://www.tiktok.com/@mindfacts",
]

VIDEO_POSITIONS = [30, 31, 32]  # skip newest 29, use videos at these positions


def _playwright_get_channel_videos(channel_url: str, positions: list) -> list:
    """
    Opens TikTok channel with headless Chromium, scrolls until the target
    video positions are loaded, visits each video page and extracts its
    description text. Returns a list of dicts with channel/position/url/text.
    """
    from playwright.sync_api import sync_playwright

    handle = channel_url.rstrip("/").split("@")[-1]
    results = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/17.0 Mobile/15E148 Safari/604.1"
                ),
                viewport={"width": 390, "height": 844},
                locale="en-US",
            )
            page = ctx.new_page()
            page.goto(channel_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)

            # Scroll until we have at least max(positions)+3 video links loaded
            target = max(positions) + 3
            for _ in range(30):
                links = page.locator('a[href*="/video/"]').all()
                if len(links) >= target:
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1800)

            links = page.locator('a[href*="/video/"]').all()
            print(f"[{handle}] {len(links)} video links loaded")

            for pos in positions:
                idx = pos - 1
                if idx >= len(links):
                    print(f"[{handle}] only {len(links)} links, can't reach #{pos}")
                    continue

                href = links[idx].get_attribute("href") or ""
                if not href:
                    continue
                if href.startswith("/"):
                    href = f"https://www.tiktok.com{href}"

                # Visit the video page to extract its description / caption text
                vp = ctx.new_page()
                try:
                    vp.goto(href, wait_until="domcontentloaded", timeout=20000)
                    vp.wait_for_timeout(2500)

                    text = ""
                    for sel in [
                        '[data-e2e="video-desc"]',
                        '[data-e2e="browse-video-desc"]',
                        'h1[data-e2e]',
                        'span[class*="SpanText"]',
                        "h1",
                    ]:
                        try:
                            t = vp.locator(sel).first.inner_text(timeout=2000).strip()
                            if t and len(t) > 5:
                                text = t[:500]
                                break
                        except Exception:
                            pass

                    results.append({"channel": handle, "position": pos, "url": href, "text": text})
                    print(f"[{handle}] #{pos}: {text[:80]}")
                except Exception as e:
                    results.append({"channel": handle, "position": pos, "url": href, "text": "", "error": str(e)})
                finally:
                    vp.close()

            browser.close()

    except Exception as e:
        print(f"Playwright error for {channel_url}: {e}")

    return results


@app.route("/suggest-topics", methods=["GET"])
def suggest_topics():
    """
    Uses Playwright to visit each inspiration channel, fetches the video at
    positions 30, 31, 32 (3 channels × 3 positions = 9 videos). Returns 9
    topic ideas — one per video — so the user can verify by counting down
    to those exact positions on each channel.
    """
    all_videos = []
    for account_url in INSPIRATION_ACCOUNTS:
        videos = _playwright_get_channel_videos(account_url, VIDEO_POSITIONS)
        all_videos.extend(videos)

    if not all_videos:
        # Playwright unavailable or all channels blocked — Claude fallback
        import time as _t
        prompt = f"""You are a TikTok science content strategist for @provenweird.

Generate 9 ORIGINAL topic ideas for 60-second science/facts videos. Cover biology, physics, psychology, space, ocean, human body, animals, historical paradoxes. Each topic must have a built-in paradox, counterintuitive angle, or philosophical hook. No "Did you know". Vary the formula. Each phrase 10-20 words max.

Seed: {int(_t.time())}

Return ONLY a JSON array of 9 strings. No markdown."""
        source = "claude_fallback"
    else:
        lines = [
            f"@{v['channel']} video #{v['position']}: {v['text'] or '(no caption extracted)'}"
            for v in all_videos
        ]
        sample = "\n".join(lines)
        prompt = f"""You are a TikTok science content strategist for @provenweird.

Below are the actual captions/descriptions of {len(all_videos)} videos at positions 30–32 on 3 successful science facts channels. These are proven topics from ~3–6 months ago.

For EACH video listed, generate ONE inspired but wholly original topic idea that captures the same energy and angle but uses a completely different fact. Output exactly {len(all_videos)} topics in the same order so they can be mapped back to each video.

Videos:
{sample}

Rules: no "Did you know", each topic 10-20 words max, dry sardonic tone, vary the formula.

Return ONLY a JSON array of {len(all_videos)} strings. No markdown. No explanation."""
        source = "playwright"

    resp = http_requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
        raw = raw.rsplit("```", 1)[0].strip()
    topics = json.loads(raw)

    return jsonify({
        "topics": topics,
        "source": source,
        "source_count": len(all_videos),
        "videos": all_videos,   # channel + position + url + text — for verification
    })


# ─── health ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/debug/config")
def debug_config():
    return jsonify({
        "client_key_prefix": TIKTOK_CLIENT_KEY[:8] if TIKTOK_CLIENT_KEY else "NOT_SET",
        "scope": TIKTOK_SCOPE,
        "render_url": RENDER_URL,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
