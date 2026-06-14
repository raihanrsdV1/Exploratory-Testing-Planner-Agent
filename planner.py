import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from litellm import completion
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip() or None

if not LLM_PROVIDER:
    raise RuntimeError("Missing LLM_PROVIDER in .env")

if not LLM_MODEL:
    raise RuntimeError("Missing LLM_MODEL in .env")

if not LLM_API_KEY:
    raise RuntimeError("Missing LLM_API_KEY in .env")


def configure_provider_env() -> None:
    """
    LiteLLM reads provider-specific API keys from environment variables.
    This function maps one generic LLM_API_KEY to the correct provider key.
    """

    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "together": "TOGETHERAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cohere": "COHERE_API_KEY",
        "xai": "XAI_API_KEY",
    }

    env_name = provider_key_map.get(LLM_PROVIDER)

    if not env_name:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{LLM_PROVIDER}'. "
            "Add it to provider_key_map first."
        )

    os.environ[env_name] = LLM_API_KEY

    if LLM_API_BASE:
        os.environ["OPENAI_API_BASE"] = LLM_API_BASE


configure_provider_env()


app = FastAPI(title="Generic LLM API")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: Optional[str] = Field(
        default=(
            "You are a helpful software testing and coding assistant. "
            "Give clear, practical, structured answers. "
            "When asked to return JSON, return valid JSON only."
        )
    )
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


def extract_text(response: Any) -> str:
    """
    LiteLLM returns an OpenAI-like response.
    Usually: response.choices[0].message.content
    """

    try:
        content = response.choices[0].message.content
    except Exception:
        content = response["choices"][0]["message"]["content"]

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()


def generate_answer(
    prompt: str,
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    if LLM_API_BASE:
        kwargs["api_base"] = LLM_API_BASE

    response = completion(**kwargs)
    return extract_text(response)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "api_base": LLM_API_BASE,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        answer = generate_answer(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

        return {
            "prompt": req.prompt,
            "answer": answer,
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(error)}",
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )