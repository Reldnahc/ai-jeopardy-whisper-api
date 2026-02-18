import base64
import os
import tempfile
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Form
from pydantic import BaseModel
from faster_whisper import WhisperModel

APP_MODEL = os.getenv("WHISPER_MODEL", "medium")
DEVICE = os.getenv("DEVICE", "cuda")          # "cuda" or "cpu"
COMPUTE = os.getenv("COMPUTE_TYPE", "float16") # "float16", "int8_float16", "int8"
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))

app = FastAPI(title="Whisper STT API", version="1.0")

# Load once at startup
model = WhisperModel(APP_MODEL, device=DEVICE, compute_type=COMPUTE)

class Base64Request(BaseModel):
    audio_base64: str  # raw base64 string (no data: prefix)
    filename: Optional[str] = "audio.wav"
    language: Optional[str] = None
    task: str = "transcribe"  # "transcribe" or "translate"

    # NEW:
    prompt: Optional[str] = None
    temperature: Optional[float] = None

def _save_temp_bytes(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path

def _transcribe(
    path: str,
    language: Optional[str],
    task: str,
    prompt: Optional[str],
    temperature: Optional[float],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "language": language,
        "task": task,
        "beam_size": BEAM_SIZE,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 250},
    }

    # NEW: dynamic initial prompt
    if prompt and prompt.strip():
        kwargs["initial_prompt"] = prompt.strip()

    # Optional: allow caller to tweak temperature
    if temperature is not None:
        # keep it sane
        t = float(temperature)
        if t < 0 or t > 1.0:
            raise HTTPException(status_code=400, detail="temperature must be between 0 and 1.0")
        kwargs["temperature"] = t

    segments, info = model.transcribe(path, **kwargs)

    segs: List[Dict[str, Any]] = []
    full_text_parts = []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text})
        full_text_parts.append(s.text)

    return {
        "text": "".join(full_text_parts).strip(),
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segs,
    }

@app.get("/health")
def health():
    return {"ok": True, "model": APP_MODEL, "device": DEVICE, "compute_type": COMPUTE}

@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),

    # NEW:
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
):
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail="task must be transcribe or translate")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    path = _save_temp_bytes(data, suffix=suffix)
    try:
        return _transcribe(path, language, task, prompt, temperature)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

@app.post("/transcribe_base64")
def transcribe_base64(payload: Base64Request = Body(...)):
    if payload.task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail="task must be transcribe or translate")

    try:
        cleaned = "".join(payload.audio_base64.split())
        raw = base64.b64decode(cleaned)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64")

    suffix = os.path.splitext(payload.filename or "")[1] or ".wav"
    path = _save_temp_bytes(raw, suffix=suffix)
    try:
        return _transcribe(path, payload.language, payload.task, payload.prompt, payload.temperature)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
