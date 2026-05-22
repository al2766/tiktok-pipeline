"""
Flask API wrapper around the TikTok video pipeline.
POST /generate         { "topic": "..." }               → { "job_id": "..." }
GET  /status/:id                                        → { "status": "...", "video_b64": "...", "caption": "..." }
POST /post             { "job_id": "...", "title": "..." } → { "ok": true, "publish_id": "..." }
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

from pipeline import generate_video, generate_script

app = Flask(__name__)
CORS(app)

jobs: dict = {}   # job_id → { status, video_b64, caption, error }

TIKTOK_CLIENT_KEY    = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
RENDER_URL           = "https://tiktok-pipeline-cgtt.onrender.com"
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
        script_data = generate_script(topic)
        jobs[job_id]["caption"] = script_data.get("tiktok_caption", "")

        from pipeline import generate_image, generate_tts, assemble_video
        import re, time
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(time.time())}"

        image_path = generate_image(script_data["visual_prompt"], slug)
        audio_path = generate_tts(script_data["script"], slug)
        video_path = assemble_video(image_path, audio_path, script_data, slug)

        with open(video_path, "rb") as f:
            jobs[job_id]["video_b64"] = base64.b64encode(f.read()).decode()

        image_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        jobs[job_id]["status"] = "done"
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
        "scope": "video.publish,user.info.basic",
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


# ─── health ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
