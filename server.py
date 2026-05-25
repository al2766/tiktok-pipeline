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

# Instagram channels — Playwright visits these, fetches positions 9/10/11
# (Instagram shows ~12 without auth so 9-11 are reliably reachable)
INSPIRATION_ACCOUNTS = [
    "https://www.instagram.com/explainingtheocean/reels/",
    "https://www.instagram.com/succezzcity/reels/",
    "https://www.instagram.com/geopandamaps/reels/",
]
VIDEO_POSITIONS = [9, 10, 11]

# Reddit book research — User-Agent required or Reddit blocks the request
_REDDIT_UA = "Mozilla/5.0 (compatible; provenweird-bot/1.0 +https://provenweird.com)"


def _reddit_get_trending_books() -> list[str]:
    """
    Queries r/Fantasy top posts (last month) and asks Claude to extract
    the most-discussed book/series names. Falls back to a curated seed list.
    """
    try:
        r = http_requests.get(
            "https://www.reddit.com/r/Fantasy/top.json",
            params={"t": "month", "limit": 25},
            headers={"User-Agent": _REDDIT_UA},
            timeout=8,
        )
        posts = r.json()["data"]["children"]
        titles = [p["data"]["title"] for p in posts[:20]]
        title_block = "\n".join(titles)

        resp = http_requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "messages": [{"role": "user", "content": (
                    "From these Reddit r/Fantasy post titles, extract the names of the "
                    "3 most-discussed specific books or series. Return a JSON array of strings only.\n\n"
                    + title_block
                )}],
            },
            timeout=10,
        )
        raw = resp.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Reddit book fetch failed: {e}")
        return ["ACOTAR", "Fourth Wing", "The Name of the Wind"]


def _reddit_get_book_moments(book: str) -> list[str]:
    """
    Searches Reddit for memorable/emotional scenes from a book.
    Uses multiple search terms to catch threads that phrase it differently.
    Returns up to 4 post titles/excerpts.
    """
    results = []
    search_terms = [
        f"{book} scene emotional",
        f"{book} that moment chapter",
        f"{book} plot twist reaction",
        f'"{book}" favorite scene',
    ]
    tried = set()
    for terms in search_terms:
        if len(results) >= 4:
            break
        if terms in tried:
            continue
        tried.add(terms)
        try:
            r = http_requests.get(
                "https://www.reddit.com/search.json",
                params={"q": terms, "sort": "top", "t": "year", "limit": 5},
                headers={"User-Agent": _REDDIT_UA},
                timeout=8,
            )
            for p in r.json()["data"]["children"]:
                d = p["data"]
                snippet = (d.get("selftext") or "")[:150].strip()
                entry = d["title"] + (f" — {snippet}" if snippet else "")
                results.append(entry[:250])
                if len(results) >= 4:
                    break
        except Exception:
            pass
    return results


def _reddit_get_psychology_topics() -> list[str]:
    """
    Gets top posts from r/psychology and r/selfimprovement this month.
    Returns titles of the most upvoted discussions — raw material for
    psychology hack hooks.
    """
    results = []
    for sub in ["psychology", "selfimprovement"]:
        try:
            r = http_requests.get(
                f"https://www.reddit.com/r/{sub}/top.json",
                params={"t": "month", "limit": 10},
                headers={"User-Agent": _REDDIT_UA},
                timeout=8,
            )
            for p in r.json()["data"]["children"]:
                results.append(p["data"]["title"][:200])
        except Exception:
            pass
    return results[:12]


def _playwright_get_channel_videos(channel_url: str, positions: list) -> list:
    """
    Opens TikTok channel with headless Chromium, scrolls until the target
    video positions are loaded, visits each video page and extracts its
    description text. Returns a list of dicts with channel/position/url/text.
    """
    from playwright.sync_api import sync_playwright

    handle = channel_url.rstrip("/").split("@")[-1].split("/")[0]
    results = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = ctx.new_page()
            page.goto(channel_url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(4000)

            # Dismiss Instagram login popup
            for sel in ['[aria-label="Close"]', 'button:has-text("Not now")', 'button:has-text("Not Now")']:
                try:
                    page.locator(sel).first.click(timeout=2000)
                    page.wait_for_timeout(1000)
                except Exception:
                    pass

            # Scroll to trigger lazy load
            target = max(positions) + 3
            for _ in range(8):
                links = page.locator('a[href*="/reel/"], a[href*="/p/"]').all()
                hrefs_raw = [l.get_attribute("href") for l in links]
                hrefs_raw = [h for h in hrefs_raw if h and ("/reel/" in h or "/p/" in h)]
                seen: set = set()
                unique = [h for h in hrefs_raw if not (h in seen or seen.add(h))]  # type: ignore[func-returns-value]
                if len(unique) >= target:
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1500)

            print(f"[{handle}] {len(unique)} reel links loaded")

            for pos in positions:
                idx = pos - 1
                if idx >= len(unique):
                    print(f"[{handle}] only {len(unique)} links, can't reach #{pos}")
                    continue

                href = unique[idx]
                if href.startswith("/"):
                    href = f"https://www.instagram.com{href}"

                vp = ctx.new_page()
                try:
                    vp.goto(href, wait_until="networkidle", timeout=20000)
                    vp.wait_for_timeout(3000)
                    for sel in ['[aria-label="Close"]', 'button:has-text("Not now")']:
                        try:
                            vp.locator(sel).first.click(timeout=1500)
                        except Exception:
                            pass

                    text = ""
                    # og:description is most reliable on Instagram
                    try:
                        text = vp.get_attribute('meta[property="og:description"]', "content", timeout=3000) or ""
                    except Exception:
                        pass
                    if not text:
                        for sel in ["h1", '[class*="_a9zs"]', 'span[class*="caption"]']:
                            try:
                                t = vp.locator(sel).first.inner_text(timeout=2000).strip()
                                if t and len(t) > 5:
                                    text = t
                                    break
                            except Exception:
                                pass
                    text = text[:400]
                    results.append({"channel": handle, "position": pos, "url": href, "text": text})
                    print(f"[{handle}] #{pos}: {text[:100]}")
                except Exception as e:
                    results.append({"channel": handle, "position": pos, "url": href, "text": "", "error": str(e)})
                finally:
                    vp.close()

            browser.close()

    except Exception as e:
        print(f"Playwright error for {channel_url}: {e}")

    return results


def _call_claude(prompt: str, max_tokens: int = 800) -> str:
    resp = http_requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


@app.route("/suggest-topics", methods=["GET"])
def suggest_topics():
    """
    Generates 9 topic ideas across 3 formats, each grounded in real research:

    FORMAT 1 — Scenario Game (3 topics)
    Real Instagram channel captions (positions 9-11) → viewer-inside-a-scenario hooks.
    The fact plays out like a game. Viewer picks a side or imagines they're there.

    FORMAT 2 — Book/Famous Moment Proxy (3 topics)
    Reddit r/Fantasy trending books → memorable emotional scenes → map to real
    psychology or biology fact → @provenweird explains the real science through
    the book moment. Book readers self-select immediately.

    FORMAT 3 — Psychology Hack (3 topics)
    Reddit r/psychology + r/selfimprovement top posts → reframe as a usable
    human behaviour insight. Dry, not preachy. Teaches through specificity.
    """
    import threading, time as _t

    # ── Run all 3 research sources in parallel ─────────────────────────────────
    insta_videos: list = []
    book_data: dict = {"books": [], "moments": {}}
    psych_titles: list = []

    def fetch_instagram():
        for url in INSPIRATION_ACCOUNTS:
            insta_videos.extend(_playwright_get_channel_videos(url, VIDEO_POSITIONS))

    def fetch_books():
        books = _reddit_get_trending_books()
        book_data["books"] = books
        for book in books[:3]:
            book_data["moments"][book] = _reddit_get_book_moments(book)

    def fetch_psych():
        psych_titles.extend(_reddit_get_psychology_topics())

    threads = [
        threading.Thread(target=fetch_instagram, daemon=True),
        threading.Thread(target=fetch_books, daemon=True),
        threading.Thread(target=fetch_psych, daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    seed = int(_t.time())

    # ── Build the prompt with all 3 research inputs ────────────────────────────

    # Format 1: scenario game — from Instagram channel data
    if insta_videos:
        insta_block = "\n".join(
            f"@{v['channel']} #{v['position']}: {v.get('text','')[:200]}"
            for v in insta_videos
        )
        f1_instruction = f"""FORMAT 1 — SCENARIO GAME (generate exactly 3 topics)
Inspired by these real Instagram science/geography reels (positions 9-11):
{insta_block}

Each topic puts the viewer INSIDE a scenario before revealing the fact.
Opens with: "What if you were..." / "If this happened right now..." / "You're in [place], and..."
The science answers what happens next. Viewer is a character in the story.
Do NOT end with a joke or punchline. The scenario IS the hook — the fact is the payoff."""
    else:
        f1_instruction = f"""FORMAT 1 — SCENARIO GAME (generate exactly 3 topics)
Topics that put the viewer inside a physical or psychological scenario before revealing the fact.
Opens with: "What if you were..." / "If this happened to you right now..."
The science answers what happens. Viewer is a character. No punchline — the fact is the payoff.
Seed: {seed}"""

    # Format 2: book moment proxy — from Reddit book research
    books = book_data.get("books", ["ACOTAR", "Fourth Wing", "The Name of the Wind"])
    moments_lines = []
    for book, moments in book_data.get("moments", {}).items():
        for m in moments[:2]:
            moments_lines.append(f"{book}: {m}")
    if moments_lines:
        moments_block = "\n".join(moments_lines[:6])
        f2_instruction = f"""FORMAT 2 — BOOK MOMENT PROXY (generate exactly 3 topics)
Trending books on Reddit r/Fantasy right now: {', '.join(books)}
Memorable scenes/moments being discussed:
{moments_block}

Each topic names a specific book/scene in the first 5 words, then reveals the real science behind it.
Example structure: "[Book] fans — that scene where [X happens] is a real [phenomenon]."
Book readers self-select. Non-readers still learn the real fact. No spoiler framing needed — the science IS the reveal.
Use different books across the 3 topics."""
    else:
        f2_instruction = f"""FORMAT 2 — BOOK MOMENT PROXY (generate exactly 3 topics)
Pick 3 well-known fantasy/fiction books (ACOTAR, Fourth Wing, The Name of the Wind, or similar).
Each topic: name the book + a specific type of scene (fear, heartbreak, adrenaline, grief), then
reveal the real psychology/biology behind why that scene hits so hard.
Book readers self-select from the first 5 words. No spoilers needed — the science is the reveal.
Seed: {seed}"""

    # Format 3: psychology hack — from Reddit psychology data
    if psych_titles:
        psych_block = "\n".join(psych_titles[:8])
        f3_instruction = f"""FORMAT 3 — PSYCHOLOGY HACK (generate exactly 3 topics)
Top discussions on r/psychology and r/selfimprovement right now:
{psych_block}

Each topic is a real human behaviour pattern framed as something specific and usable.
NOT motivational. NOT advice. Just the fact stated in a way that makes the viewer feel smarter.
Example: "The reason you work harder after almost winning isn't motivation — it's loss aversion."
Dry, specific, no preachiness. The insight should feel like a cheat code, not a lecture."""
    else:
        f3_instruction = f"""FORMAT 3 — PSYCHOLOGY HACK (generate exactly 3 topics)
Each topic is a real human behaviour or social psychology fact framed as a specific, usable insight.
NOT motivational. NOT advice. Dry, factual, makes the viewer feel smarter.
Example: "The reason you work harder after almost winning isn't motivation — it's loss aversion."
Seed: {seed}"""

    prompt = f"""You are writing topic hooks for @provenweird — a science/facts short video channel.
Voice: dry, confident, never excited, never preachy, never trying to be funny for the sake of it.

Generate exactly 9 topic hooks: 3 per format below. Each 10-20 words.

CRITICAL RULES:
- No "Did you know"
- No punchlines that mock things people love (games, books, hobbies)
- No passive voice in the hook
- Each hook must make a viewer STOP because they're personally invested — not because it's clever
- Dry tone = confident and specific, not deadpan comedy

{f1_instruction}

{f2_instruction}

{f3_instruction}

Return ONLY a JSON object with keys "scenario_game", "book_moment", "psychology_hack" — each a list of 3 strings.
No markdown. No explanation."""

    raw = _call_claude(prompt, max_tokens=900)
    try:
        data = json.loads(raw)
        topics = (
            data.get("scenario_game", []) +
            data.get("book_moment", []) +
            data.get("psychology_hack", [])
        )
    except Exception:
        topics = []

    return jsonify({
        "topics": topics,
        "by_format": data if "data" in dir() else {},
        "source": "playwright+reddit" if (insta_videos or moments_lines) else "claude_fallback",
        "source_count": len(insta_videos),
        "videos": insta_videos,
        "books_researched": books,
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
