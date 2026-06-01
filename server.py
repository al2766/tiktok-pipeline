"""
Flask API for the @provenweird TikTok video pipeline.

POST /generate          { "topic": "..." }               → { "job_id": "..." }
GET  /status/:id                                         → { "status", "step", "caption", "error" }
GET  /download/:id                                       → video/mp4 stream
POST /post              { "job_id": "...", "title": "..." } → { "ok": true, "publish_id": "..." }
GET  /oauth/start                                        → redirect to TikTok OAuth
GET  /oauth/callback                                     → exchange code, store tokens
GET  /oauth/status                                       → { "authorized": true/false }
GET  /suggest-topics                                     → { "topics": [...] }
GET  /health
"""

import os, uuid, threading, json, re, time as _time
from pathlib import Path
from flask import Flask, request, jsonify, redirect, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import requests as http_requests

load_dotenv()

import imageio_ffmpeg
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

from pipeline_v6 import generate_script, generate_tts, generate_images, animate_images, assemble_clips, upload_to_drive

app  = Flask(__name__)
CORS(app)

jobs: dict = {}  # job_id → { status, step, video_path, caption, error }

TIKTOK_CLIENT_KEY    = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
RENDER_URL           = os.environ.get("RENDER_URL", "https://tiktok-pipeline-cgtt.onrender.com")
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


# ─── Pipeline job ─────────────────────────────────────────────────────────────

def _run_job(job_id: str, topic: str):
    jobs[job_id]["status"] = "generating"
    try:
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40] + f"_{int(_time.time())}"

        jobs[job_id]["step"] = "script"
        script_data = generate_script(topic)
        jobs[job_id]["caption"] = script_data.get("tiktok_caption", "")

        jobs[job_id]["step"] = "voice"
        audio_path, word_boundaries = generate_tts(script_data["script"], slug)

        jobs[job_id]["step"] = "images"
        image_paths = generate_images(script_data["visual_prompts"], slug)

        jobs[job_id]["step"] = "animation"
        clip_paths = animate_images(image_paths, slug)

        jobs[job_id]["step"] = "assembly"
        video_path = assemble_clips(clip_paths, audio_path, word_boundaries, slug, cleanup_images=image_paths)

        jobs[job_id]["video_path"] = str(video_path)

        jobs[job_id]["step"] = "uploading"
        jobs[job_id]["drive_url"] = upload_to_drive(video_path)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["step"]   = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        print(f"Job {job_id} failed: {e}")


# ─── Generate ─────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    data  = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic required"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "step": "", "video_path": None, "caption": None, "drive_url": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, topic), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify({k: v for k, v in job.items() if k != "video_path"})


@app.route("/download/<job_id>")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "not ready"}), 404
    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "video file not found"}), 404
    return send_file(video_path, mimetype="video/mp4", as_attachment=False)


# ─── TikTok OAuth ─────────────────────────────────────────────────────────────

@app.route("/oauth/start")
def oauth_start():
    from urllib.parse import urlencode
    params = {
        "client_key":    TIKTOK_CLIENT_KEY,
        "response_type": "code",
        "scope":         TIKTOK_SCOPE,
        "redirect_uri":  f"{RENDER_URL}/oauth/callback",
        "state":         "provenweird",
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
            "client_key":    TIKTOK_CLIENT_KEY,
            "client_secret": TIKTOK_CLIENT_SECRET,
            "code":          code,
            "grant_type":    "authorization_code",
            "redirect_uri":  f"{RENDER_URL}/oauth/callback",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    token_data = resp.json()
    if "access_token" not in token_data:
        return f"<h2>Token exchange failed</h2><pre>{token_data}</pre>", 400

    _save_tokens(token_data)
    return "<h2>TikTok authorized ✓</h2><p>You can close this tab.</p>"


@app.route("/oauth/status")
def oauth_status():
    return jsonify({"authorized": _load_tokens() is not None})


# ─── Post to TikTok ───────────────────────────────────────────────────────────

@app.route("/post", methods=["POST"])
def post_to_tiktok():
    data   = request.get_json(force=True)
    job_id = data.get("job_id", "")
    title  = (data.get("title") or "").strip() or "Mind-blowing science fact #science #facts #learnontiktok"

    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Job not found or not complete"}), 400

    tokens = _load_tokens()
    if not tokens:
        return jsonify({"error": "not_authorized", "auth_url": f"{RENDER_URL}/oauth/start"}), 401

    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found on server"}), 500

    access_token = tokens["access_token"]
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_size = len(video_bytes)

    init_resp = http_requests.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json; charset=UTF-8"},
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
                "source":            "FILE_UPLOAD",
                "video_size":        video_size,
                "chunk_size":        video_size,
                "total_chunk_count": 1,
            },
        },
    )
    init_data = init_resp.json()
    if init_data.get("error", {}).get("code", "").lower() != "ok":
        return jsonify({"error": "TikTok init failed", "detail": init_data}), 500

    publish_id = init_data["data"]["publish_id"]
    upload_url = init_data["data"]["upload_url"]

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


# ─── Topic suggestions ────────────────────────────────────────────────────────

INSPIRATION_ACCOUNTS = [
    "https://www.instagram.com/explainingtheocean/reels/",
    "https://www.instagram.com/succezzcity/reels/",
    "https://www.instagram.com/geopandamaps/reels/",
]
VIDEO_POSITIONS = [9, 10, 11]
_REDDIT_UA = "Mozilla/5.0 (compatible; provenweird-bot/1.0)"


def _reddit_get_trending_books() -> list[str]:
    try:
        r = http_requests.get(
            "https://www.reddit.com/r/Fantasy/top.json",
            params={"t": "month", "limit": 25},
            headers={"User-Agent": _REDDIT_UA},
            timeout=8,
        )
        titles = [p["data"]["title"] for p in r.json()["data"]["children"][:20]]
        resp = _call_claude(
            "From these Reddit r/Fantasy post titles, extract the names of the "
            "3 most-discussed specific books or series. Return a JSON array of strings only.\n\n"
            + "\n".join(titles),
            max_tokens=150,
        )
        return json.loads(resp)
    except Exception as e:
        print(f"Reddit book fetch failed: {e}")
        return ["ACOTAR", "Fourth Wing", "The Name of the Wind"]


def _reddit_get_book_moments(book: str) -> list[str]:
    results = []
    for terms in [f"{book} scene emotional", f"{book} plot twist reaction", f'"{book}" favorite scene']:
        if len(results) >= 4:
            break
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
                results.append((d["title"] + (f" — {snippet}" if snippet else ""))[:250])
                if len(results) >= 4:
                    break
        except Exception:
            pass
    return results


def _reddit_get_psychology_topics() -> list[str]:
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
    from playwright.sync_api import sync_playwright
    handle  = channel_url.rstrip("/").split("@")[-1].split("/")[0]
    results = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = ctx.new_page()
            page.goto(channel_url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(4000)
            for sel in ['[aria-label="Close"]', 'button:has-text("Not now")', 'button:has-text("Not Now")']:
                try:
                    page.locator(sel).first.click(timeout=2000)
                    page.wait_for_timeout(1000)
                except Exception:
                    pass
            target = max(positions) + 3
            unique: list = []
            for _ in range(8):
                links     = page.locator('a[href*="/reel/"], a[href*="/p/"]').all()
                hrefs_raw = [l.get_attribute("href") for l in links]
                hrefs_raw = [h for h in hrefs_raw if h and ("/reel/" in h or "/p/" in h)]
                seen: set = set()
                unique    = [h for h in hrefs_raw if not (h in seen or seen.add(h))]  # type: ignore[func-returns-value]
                if len(unique) >= target:
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1500)
            print(f"[{handle}] {len(unique)} reel links")
            for pos in positions:
                idx = pos - 1
                if idx >= len(unique):
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
                    results.append({"channel": handle, "position": pos, "url": href, "text": text[:400]})
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
        raw = raw.split("```")[1].lstrip("json").strip().rsplit("```", 1)[0].strip()
    return raw


@app.route("/suggest-topics", methods=["GET"])
def suggest_topics():
    insta_videos: list = []
    book_data: dict    = {"books": [], "moments": {}}
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
        threading.Thread(target=fetch_books,     daemon=True),
        threading.Thread(target=fetch_psych,     daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    seed = int(_time.time())
    books = book_data.get("books", ["ACOTAR", "Fourth Wing", "The Name of the Wind"])

    if insta_videos:
        insta_block = "\n".join(f"@{v['channel']} #{v['position']}: {v.get('text','')[:200]}" for v in insta_videos)
        f1 = f"""FORMAT 1 — SCENARIO GAME (exactly 3 topics)
Inspired by these Instagram science/geography reels:
{insta_block}
Put the viewer INSIDE a scenario. Opens with "What if you were..." / "If this happened right now..."
The science answers what happens next. No punchline — the fact is the payoff."""
    else:
        f1 = f"""FORMAT 1 — SCENARIO GAME (exactly 3 topics)
Put the viewer inside a physical or psychological scenario before revealing the fact.
Opens with "What if you were..." / "If this happened to you right now..."
Seed: {seed}"""

    moments_lines = []
    for book, moments in book_data.get("moments", {}).items():
        for m in moments[:2]:
            moments_lines.append(f"{book}: {m}")
    if moments_lines:
        f2 = f"""FORMAT 2 — BOOK MOMENT PROXY (exactly 3 topics)
Trending books: {', '.join(books)}
Discussed scenes:
{chr(10).join(moments_lines[:6])}
Name the book in first 5 words, reveal the real science behind the scene.
Example: "[Book] fans — that scene where [X] is real [phenomenon]."
Use different books across the 3 topics."""
    else:
        f2 = f"""FORMAT 2 — BOOK MOMENT PROXY (exactly 3 topics)
Pick 3 well-known fantasy/fiction books. Name the book + a scene type (fear, heartbreak, adrenaline),
then reveal the real psychology/biology behind why that scene hits.
Seed: {seed}"""

    if psych_titles:
        f3 = f"""FORMAT 3 — PSYCHOLOGY HACK (exactly 3 topics)
Top Reddit r/psychology + r/selfimprovement posts:
{chr(10).join(psych_titles[:8])}
Each topic: real human behaviour pattern framed as something specific and usable.
NOT motivational. NOT advice. Dry, specific, makes the viewer feel smarter.
Example: "The reason you work harder after almost winning isn't motivation — it's loss aversion." """
    else:
        f3 = f"""FORMAT 3 — PSYCHOLOGY HACK (exactly 3 topics)
Real human behaviour or social psychology fact, framed as a specific usable insight.
Dry, factual, no preachiness. Makes the viewer feel smarter.
Seed: {seed}"""

    prompt = f"""You are writing topic hooks for @provenweird — a science/facts short video channel.
Voice: dry, confident, never excited, never preachy.

Generate exactly 9 topic hooks: 3 per format. Each 10-20 words.

RULES:
- No "Did you know"
- No passive voice in the hook
- Each hook makes a viewer STOP because they're personally invested

{f1}

{f2}

{f3}

Return ONLY a JSON object: {{"scenario_game": [...3...], "book_moment": [...3...], "psychology_hack": [...3...]}}
No markdown. No explanation."""

    raw = _call_claude(prompt, max_tokens=900)
    try:
        data   = json.loads(raw)
        topics = data.get("scenario_game", []) + data.get("book_moment", []) + data.get("psychology_hack", [])
    except Exception:
        data, topics = {}, []

    return jsonify({
        "topics":           topics,
        "by_format":        data,
        "source":           "playwright+reddit" if (insta_videos or moments_lines) else "claude_fallback",
        "books_researched": books,
    })


# ─── Topic queue (Reddit scrape + Claude scoring) ────────────────────────────

_SCORE_PROMPT = """You pick topics for @provenweird — a TikTok channel making 25-second fact videos.

The winning formula: an everyday object or animal the viewer has definitely seen before, explained in 25 seconds, with one surprising truth they never knew. The hook is always "most people think X, but actually Y."

Best examples of topics that work:
- Why escalator brushes are designed to annoy you (not clean shoes)
- Why the hole in a soda tab exists (not for a straw)
- Why pandas are forced to watch TV at the zoo
- Why belugas are considered the kindest animal on earth
- Why cats only meow at humans and never at other cats

From the Reddit posts below, pick the 5 best topics. For each return:
- clean_topic: 6-12 word topic phrase for a script generator (NOT the Reddit title — reframe it as a "why/what/how" question or surprising statement)
- familiarity: 0-10 (has every viewer definitely seen this thing in real life?)
- visual: 0-10 (can the whole video show ONE consistent scene — same animal, object, or place?)
- length: 0-10 (fully explainable in 25 seconds?)
- total: sum of the three scores

Skip anything: NSFW, about specific living people, political, requiring more than 30 seconds, or too obscure for a general audience.

Return ONLY a valid JSON array of exactly 5 objects. No markdown, no explanation.

Reddit posts:
{posts}"""


@app.route("/topics/refresh")
def topics_refresh():
    subreddits = [
        ("todayilearned",    500, 30),
        ("interestingasfuck", 300, 20),
        ("mildlyinteresting", 200, 15),
    ]

    candidates = []

    def _fetch(sub, min_score, limit):
        try:
            r = http_requests.get(
                f"https://www.reddit.com/r/{sub}/top.json",
                params={"t": "day", "limit": limit},
                headers={"User-Agent": _REDDIT_UA},
                timeout=8,
            )
            for post in r.json()["data"]["children"]:
                d = post["data"]
                if (d["score"] >= min_score
                        and not d.get("over_18")
                        and len(d["title"]) > 20):
                    candidates.append({
                        "subreddit": f"r/{sub}",
                        "title":     d["title"][:220],
                        "upvotes":   d["score"],
                        "url":       f"https://reddit.com{d['permalink']}",
                    })
        except Exception as e:
            print(f"Reddit fetch {sub} failed: {e}")

    threads = [
        threading.Thread(target=_fetch, args=(sub, ms, lim), daemon=True)
        for sub, ms, lim in subreddits
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=12)

    if not candidates:
        return jsonify({"error": "No Reddit posts fetched"}), 500

    # Deduplicate and take top 30 by upvotes
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: x["upvotes"], reverse=True):
        key = c["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
        if len(unique) >= 30:
            break

    posts_block = "\n".join(
        f"{i+1}. [{c['subreddit']}] {c['title']}"
        for i, c in enumerate(unique)
    )

    raw = _call_claude(
        _SCORE_PROMPT.format(posts=posts_block),
        max_tokens=800,
    )

    try:
        scored = json.loads(raw)
        # Attach original metadata
        title_map = {c["title"][:60].lower(): c for c in unique}
        for item in scored:
            # Try to find the matching original post for the URL
            item["url"] = ""
            item["upvotes"] = 0
            for orig in unique:
                if any(word in orig["title"].lower()
                       for word in item.get("clean_topic", "").lower().split()[:3]):
                    item["url"]     = orig["url"]
                    item["upvotes"] = orig["upvotes"]
                    item["subreddit"] = orig["subreddit"]
                    break
        return jsonify({"topics": scored})
    except Exception as e:
        return jsonify({"error": f"Scoring failed: {e}", "raw": raw}), 500


# ─── Health / debug ───────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/debug/config")
def debug_config():
    return jsonify({
        "client_key_prefix": TIKTOK_CLIENT_KEY[:8] if TIKTOK_CLIENT_KEY else "NOT_SET",
        "scope":      TIKTOK_SCOPE,
        "render_url": RENDER_URL,
    })


@app.route("/debug/memory")
def debug_memory():
    import resource, platform
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024 / 1024 if platform.system() == "Darwin" else rss_kb / 1024
    return jsonify({"rss_mb": round(rss_mb, 1), "active_jobs": len(jobs)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
