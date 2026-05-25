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
from reel_analyzer import analyze_reel

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

# Proven science/facts channels — ordered by similarity to @provenweird.
# Strategy: skip their most recent 30 videos (competing timeframe).
# Scrape items 31–55: proven topics from ~3–6 months ago, zero overlap risk.
INSPIRATION_ACCOUNTS = [
    "https://www.tiktok.com/@scitrendz",      # science trends / nature facts
    "https://www.tiktok.com/@factsverse",     # science / history oddities
    "https://www.tiktok.com/@mindfacts",      # psychology / mind facts
    "https://www.tiktok.com/@theweirdworld",  # weird science / history
]

# Skip this many recent videos (their newest content = competing timeframe)
_SKIP_RECENT = 30
_FETCH_COUNT = 25  # how many older videos to pull per account

@app.route("/suggest-topics", methods=["GET"])
def suggest_topics():
    """
    Fetches older video titles from proven science channels (skips their newest
    _SKIP_RECENT videos to avoid topic overlap), then uses Claude Haiku to
    generate 8 inspired but original topic suggestions.
    """
    import subprocess, shutil

    yt_dlp = shutil.which("yt-dlp") or "yt-dlp"
    raw_titles = []

    playlist_range = f"{_SKIP_RECENT + 1}:{_SKIP_RECENT + _FETCH_COUNT + 1}"

    for account_url in INSPIRATION_ACCOUNTS[:3]:  # limit to 3 accounts per call
        try:
            result = subprocess.run(
                [yt_dlp, "--flat-playlist", "--playlist-items", playlist_range,
                 "--print", "%(title)s|||%(description)s",
                 "--no-warnings", "--quiet", account_url],
                capture_output=True, text=True, timeout=25
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    raw_titles.append(line.strip()[:200])
        except Exception:
            pass

    if not raw_titles:
        # Fallback: return hardcoded seed topics if scraping fails
        return jsonify({"topics": [
            "the mantis shrimp can see 16 colours humans can only see 3",
            "your body replaces most of its cells every few years",
            "honey found in ancient Egyptian tombs is still edible",
            "the human eye has a blind spot your brain hides from you",
            "tardigrades can survive in the vacuum of space",
            "cleopatra lived closer in time to the moon landing than to the pyramids",
            "a day on venus is longer than a year on venus",
            "the loudest animal on earth is smaller than your thumb",
        ]})

    sample = "\n".join(raw_titles[:30])
    prompt = f"""You are a TikTok science content strategist. Below are recent video titles and captions from successful science facts channels.

Study the topics, angles, and hooks. Then generate 8 ORIGINAL topic ideas inspired by this style — NOT copies, but new topics using the same energy and angles. Each should be a single punchy topic phrase (10-20 words max) that would work as a @provenweird video.

Focus on: biology, physics, psychology, space, ocean, human body, animal facts, historical paradoxes.

Source content:
{sample}

Return ONLY a JSON array of 8 strings. No markdown. No explanation."""

    resp = http_requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
        raw = raw.rsplit("```", 1)[0].strip()
    topics = json.loads(raw)
    return jsonify({"topics": topics, "source_count": len(raw_titles)})


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
