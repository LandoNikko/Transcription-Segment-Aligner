import os
import re
import shutil
import subprocess
import tempfile
import traceback
import unicodedata
from typing import List, Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import librosa
import numpy as np
import torch
import whisperx
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALIGN_LANGUAGE_CODES = sorted(set(DEFAULT_ALIGN_MODELS_TORCH) | set(DEFAULT_ALIGN_MODELS_HF))

# Request codes that map to another WhisperX align model (same script / tokenizer family).
# Japanese (Romaji): Latin transcript → use English aligner for word-level tokens; ja model is CJK per character.
ALIGN_MODEL_ALIASES: Dict[str, str] = {
    "ja_romaji": "en",
}

ALIGN_LANGUAGE_LABELS = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "ko": "Korean",
    "lv": "Latvian",
    "ml": "Malayalam",
    "nl": "Dutch",
    "no": "Norwegian (Bokmål)",
    "nn": "Norwegian (Nynorsk)",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "te": "Telugu",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ja_romaji": "Japanese (Romaji)",
}

API_ALIGN_LANGUAGE_CODES = sorted(set(ALIGN_LANGUAGE_CODES) | set(ALIGN_MODEL_ALIASES.keys()))

_align_label_trailing_code = re.compile(r"^(.+?)\s+\((?:[a-z]{2,3}|ja_romaji)\)\s*$", re.UNICODE)


def align_language_display_label(label: str) -> str:
    """Strip trailing locale markers like ' (en)' from dropdown labels; keep 'Norwegian (Bokmål)'."""
    s = label.strip()
    m = _align_label_trailing_code.match(s)
    return m.group(1).strip() if m else s


_align_model_cache: Dict[str, Tuple[object, dict]] = {}


def resolve_align_model_code(language_code: str) -> str:
    lc = (language_code or "en").lower().strip()
    return ALIGN_MODEL_ALIASES.get(lc, lc)


def get_align_model(language_code: str):
    lc = (language_code or "en").lower().strip()
    model_lc = resolve_align_model_code(lc)
    if model_lc not in ALIGN_LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported alignment language '{language_code}'. "
            f"Use a WhisperX align code such as 'ja' for Japanese or 'en' for English."
        )
    if model_lc not in _align_model_cache:
        print(f"Loading WhisperX alignment model for language={model_lc!r} (device={DEVICE})...")
        model_a, metadata = whisperx.load_align_model(language_code=model_lc, device=DEVICE)
        _align_model_cache[model_lc] = (model_a, metadata)
    return _align_model_cache[model_lc]


def normalize_transcript(text: str, language_code: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if (language_code or "").lower().strip() in ("ja", "zh", "ko"):
        t = unicodedata.normalize("NFKC", t)
    return t


ALLOWED_AUDIO_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma",
    ".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".mkv", ".wmv",
}


def safe_audio_suffix(original_name: str | None) -> str:
    base = os.path.basename(original_name or "") or "audio"
    ext = os.path.splitext(base)[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        ext = ".wav"
    return ext


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg was not found on PATH. WhisperX needs ffmpeg to decode audio. "
            "Install ffmpeg (e.g. https://ffmpeg.org) and restart the server."
        )


def align_transcript(audio_path: str, transcript_text: str, language_code: str):
    ensure_ffmpeg()
    lc = (language_code or "en").lower().strip()
    model_lc = resolve_align_model_code(lc)
    print(f"Loading audio {audio_path} (language={lc!r}, align_model={model_lc!r})")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file missing at {audio_path!r}")
    audio = whisperx.load_audio(audio_path)

    duration = librosa.get_duration(y=audio, sr=16000)

    transcript_segments = [
        {"text": transcript_text, "start": 0.0, "end": duration}
    ]

    model_a, metadata = get_align_model(lc)

    print("Aligning...")
    result = whisperx.align(
        transcript_segments,
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    return result


def _make_serializable(d):
    if isinstance(d, dict):
        return {k: _make_serializable(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_make_serializable(v) for v in d]
    if isinstance(d, (np.float32, np.float64, np.float16)):
        return float(d)
    return d


def generate_srt(segments: List[Dict]) -> str:
    def format_time(seconds: float) -> str:
        if seconds is None:
            return "00:00:00,000"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

    srt_content = []
    idx = 1
    for seg in segments:
        for word in seg.get('words', []):
            start = word.get('start')
            end = word.get('end')
            text = word.get('word')
            if start is not None and end is not None and text:
                srt_content.append(f"{idx}\n{format_time(start)} --> {format_time(end)}\n{text}\n")
                idx += 1
    return "\n".join(srt_content)

def generate_csv(segments: List[Dict]) -> str:
    csv_content = ["word,start,end,score"]
    for seg in segments:
        for word in seg.get('words', []):
            start = word.get('start', '')
            end = word.get('end', '')
            text = word.get('word', '').replace('"', '""')
            score = word.get('score', '')
            csv_content.append(f'"{text}",{start},{end},{score}')
    return "\n".join(csv_content)


@app.get("/api/align-languages")
def list_align_languages():
    languages = []
    for code in API_ALIGN_LANGUAGE_CODES:
        label = ALIGN_LANGUAGE_LABELS.get(code, code.upper())
        languages.append({"code": code, "label": align_language_display_label(label)})
    return {"languages": languages}


@app.post("/align")
async def api_align(
    audio: UploadFile = File(...),
    transcript: UploadFile = File(...),
    language: str = Form("en"),
):
    audio_path = None
    try:
        audio_bytes = await audio.read()
        transcript_bytes = await transcript.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload.")
        try:
            transcript_text = transcript_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Transcript must be UTF-8 text: {e}") from e

        lang = (language or "en").lower().strip()
        if lang not in API_ALIGN_LANGUAGE_CODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported language code {language!r}. "
                    f"Pick a value from GET /api/align-languages (e.g. 'ja' for Japanese, 'ja_romaji' for romaji)."
                ),
            )

        transcript_text = normalize_transcript(transcript_text, lang)

        suffix = safe_audio_suffix(audio.filename)
        fd, audio_path = tempfile.mkstemp(prefix="align_audio_", suffix=suffix)
        try:
            os.close(fd)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
        except Exception:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
            raise

        result = align_transcript(audio_path, transcript_text, lang)
        result_clean = _make_serializable(result)
        segments = result_clean.get("segments") or []
        return JSONResponse(content={
            "language": lang,
            "json": result_clean,
            "srt": generate_srt(segments),
            "csv": generate_csv(segments),
        })

    except HTTPException:
        raise
    except FileNotFoundError as e:
        if not shutil.which("ffmpeg"):
            raise HTTPException(
                status_code=500,
                detail=(
                    "ffmpeg not found on PATH (required to decode audio). "
                    "Install ffmpeg and ensure it is on your system PATH, then restart the server."
                ),
            ) from e
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ffmpeg failed to decode audio: {e}") from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if audio_path and os.path.isfile(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass

@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.get("/styles.css")
def read_styles():
    return FileResponse("styles.css", media_type="text/css")


_demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
if os.path.isdir(_demo_dir):
    app.mount("/demo", StaticFiles(directory=_demo_dir), name="demo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
