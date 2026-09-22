import os
import json
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
# Ishlab chiqarishda ALLOWED_ORIGIN ni saytingiz manziliga cheklang, masalan:
# CORS(app, resources={r"/chat": {"origins": "https://ziyokorlar.netlify.app"}})
CORS(app)

# --- Gemini sozlamalari ---
# Kalitni kodga yozmang! Render'ning "Environment Variables" bo'limiga qo'ying.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Oddiy himoya: faqat shu maxfiy so'z bilan kelgan so'rovlarga javob beradi.
SITE_SECRET = os.environ.get("SITE_SECRET", "")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:streamGenerateContent?alt=sse&key={key}"
)


def to_gemini_contents(messages):
    """OpenAI-uslubidagi messages ro'yxatini Gemini 'contents' formatiga o'giradi."""
    contents = []
    system_texts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_texts.append(content if isinstance(content, str) else "")
            continue
        gem_role = "model" if role == "assistant" else "user"
        parts = []
        if isinstance(content, str):
            text = content
            if system_texts and gem_role == "user" and not contents:
                text = "\n\n".join(system_texts) + "\n\n" + text
                system_texts = []
            parts.append({"text": text})
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        header, b64data = url.split(",", 1)
                        mime = header.split(";")[0].replace("data:", "") or "image/jpeg"
                        parts.append({"inline_data": {"mime_type": mime, "data": b64data}})
        if parts:
            contents.append({"role": gem_role, "parts": parts})
    if system_texts and contents:
        contents[0]["parts"].insert(0, {"text": "\n\n".join(system_texts)})
    return contents


@app.route("/chat", methods=["POST"])
def chat():
    if SITE_SECRET and request.headers.get("X-Site-Secret") != SITE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    if not GEMINI_API_KEY:
        return jsonify({"error": "server_not_configured"}), 500

    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "bad_request"}), 400

    payload = {
        "contents": to_gemini_contents(messages),
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.7},
    }
    url = GEMINI_URL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)

    def relay():
        try:
            with requests.post(url, json=payload, stream=True, timeout=90) as r:
                if r.status_code != 200:
                    msg = f"[Xatolik: {r.status_code}]"
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": msg}}]}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw in ("", "[DONE]"):
                        continue
                    try:
                        j = json.loads(raw)
                        cands = j.get("candidates") or []
                        text = ""
                        if cands:
                            parts = (cands[0].get("content") or {}).get("parts") or []
                            text = "".join(p.get("text", "") for p in parts)
                        if text:
                            yield "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"
                    except ValueError:
                        continue
                yield "data: [DONE]\n\n"
        except requests.RequestException:
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "[Tarmoq xatosi]"}}]}) + "\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(relay()), mimetype="text/event-stream")


@app.route("/health")
def health():
    return jsonify({"ok": True, "configured": bool(GEMINI_API_KEY)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
