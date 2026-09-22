"""
Bilimdon uchun backend — Azure OpenAI kalitini yashiradi.
Ishga tushirish: pip install flask flask-cors requests
              python app.py
Joylashtirish: Render.com yoki Railway.app kabi Python serverga (Netlify BU KODNI ishlata olmaydi).
"""
import os
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests

app = Flask(__name__)
# Ishlab chiqarishda ALLOWED_ORIGIN ni saytingiz manziliga cheklang, masalan:
# CORS(app, resources={r"/chat": {"origins": "https://bilimdon.netlify.app"}})
CORS(app)

# --- Azure OpenAI sozlamalari ---
# Kalitni kodga yozmang! Serverning "Environment Variables" bo'limiga qo'ying.
ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://mening-resursim.openai.azure.com")
API_KEY = os.environ.get("AZURE_API_KEY", "")
DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")
API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-10-21")

# Oddiy himoya: faqat shu maxfiy so'z bilan kelgan so'rovlarga javob beradi.
# Buni saytning "AI sozlamalari" qismiga backend manzili bilan birga kiritasiz.
SITE_SECRET = os.environ.get("SITE_SECRET", "")


@app.route("/chat", methods=["POST"])
def chat():
    if SITE_SECRET and request.headers.get("X-Site-Secret") != SITE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    if not API_KEY:
        return jsonify({"error": "server_not_configured"}), 500

    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "bad_request"}), 400

    url = f"{ENDPOINT.rstrip('/')}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "stream": True, "max_tokens": 2000}

    def relay():
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=90) as r:
            if r.status_code != 200:
                yield f'data: {{"choices":[{{"delta":{{"content":"[Xatolik: {r.status_code}]"}}}}]}}\n\n'
                return
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield line + "\n\n"

    return Response(stream_with_context(relay()), mimetype="text/event-stream")


@app.route("/health")
def health():
    return jsonify({"ok": True, "configured": bool(API_KEY)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)  # debug=True ni ishlab chiqarishda ishlatmang
