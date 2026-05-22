"""
Flask API wrapper around the TikTok video pipeline.
POST /generate  { "topic": "..." }  → { "job_id": "..." }
GET  /status/:id                    → { "status": "pending|generating|done|error", "video_b64": "...", "caption": "..." }
"""

import os, uuid, threading, base64
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Set bundled ffmpeg before importing pipeline
import imageio_ffmpeg
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

from pipeline import generate_video, generate_script

app = Flask(__name__)
CORS(app)

jobs: dict = {}   # job_id → { status, video_b64, caption, error }


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

        # Clean up temp files, keep video
        image_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")


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


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
