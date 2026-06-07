import os
import threading
import time
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-32b")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

print("OpenRouter model:", OPENROUTER_MODEL)
print("OPENROUTER_API_KEY configured:", bool(OPENROUTER_API_KEY))

if not OPENROUTER_API_KEY:
    print("Set OPENROUTER_API_KEY in your environment before calling /generate.")

def _split_thinking(text: str) -> dict:
    if not text:
        return {"thinking": "", "answer": ""}
    if "</think>" in text:
        thinking_content, content = text.split("</think>", 1)
        thinking_content = thinking_content.replace("<think>", "").strip()
        return {"thinking": thinking_content, "answer": content.strip()}
    return {"thinking": "", "answer": text.strip()}

def generate_answer(user_prompt: str, enable_thinking: bool = False, max_new_tokens: int = 512) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Set it in the environment first.")

    model_name = os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL)

    system_prompt = (
        "You are a precise QA test planning model. Follow the user instructions exactly. "
        "When JSON is requested, return strict valid JSON only."
    )
    if enable_thinking:
        system_prompt += " If useful, reason internally, but keep final output concise."

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Exploratory Testing Planner Agent"),
    }

    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=180)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter request failed: {resp.status_code} {resp.text[:1000]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"].get("content", "")
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc

    return _split_thinking(content)

app = FastAPI(title="OpenRouter Model API")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = 512
    enable_thinking: bool = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "openrouter",
        "model": os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL),
        "api_key_configured": bool(os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)),
    }

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        out = generate_answer(
            user_prompt=req.prompt,
            enable_thinking=req.enable_thinking,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "prompt": req.prompt,
        "answer": out["answer"],
        "thinking": out["thinking"],
    }

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    print("OpenRouter model API server started on port 8000")

    # Expose the local notebook API through ngrok (optional)
    try:
        # pyrefly: ignore [missing-import]
        from pyngrok import ngrok
        authtoken = os.getenv("NGROK_AUTHTOKEN")
        if authtoken:
            ngrok.set_auth_token(authtoken)

        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)

        tunnel = ngrok.connect(8000, "http")
        print("Public URL:", tunnel.public_url)
        print("Health check:", f"{tunnel.public_url}/health")
    except ImportError:
        print("pyngrok not installed, skipping ngrok tunnel.")

    # Wait for server to start before running smoke test
    time.sleep(2)

    # Smoke test
    base_url = "http://127.0.0.1:8000"
    payload = {
        "prompt": "Return only this JSON: {\"status\":\"ok\"}",
        "max_new_tokens": 64,
        "enable_thinking": False,
    }

    try:
        resp = requests.post(f"{base_url}/generate", json=payload, timeout=180)
        print("Smoke Test Status:", resp.status_code)
        print("Smoke Test Response:", resp.json())
    except Exception as e:
        print("Smoke Test Failed:", str(e))

    # Keep the main thread alive since the API is running in a daemon thread
    print("API server is running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
