"""
YouTube Video Pipeline
Generates a segmented script, images, voiceover, and assembles a final MP4.
Uses the new google-genai SDK (google.genai).
"""

import os
import json
import time
import wave
import subprocess
import urllib.parse
import requests
from pathlib import Path

from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SCRIPT_MODEL  = "gemini-3.1-flash-lite"
IMAGE_MODEL   = "imagen-4.0-fast-generate-001"
TTS_MODEL     = "gemini-2.5-flash-preview-tts"
IMAGE_QUOTA   = 25          # first N segments use Gemini image; rest use Pollinations
TARGET_SEGS   = 70
WORDS_PER_SEG = 22
MAX_RETRIES   = 3
RETRY_DELAY   = 5

OUTPUT_DIR = Path("output")


# ── Helpers ───────────────────────────────────────────────────────────────────

def retry(fn, *args, label="call", **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [retry {attempt}/{MAX_RETRIES}] {label} failed: {exc}. Retrying in {RETRY_DELAY}s…")
            time.sleep(RETRY_DELAY)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip().replace(" ", "_")


# ── Step 1: Script ─────────────────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    print(f"\n[SCRIPT] Generating script for: {topic}")

    prompt = (
        f"Create a YouTube video script about: {topic}\n\n"
        f"Requirements:\n"
        f"- Exactly {TARGET_SEGS} segments\n"
        f"- Each segment has ~{WORDS_PER_SEG} spoken words (narrator text)\n"
        f"- Each segment has a short, vivid visual_prompt for AI image generation\n"
        f"- Output ONLY valid JSON, no markdown fences, no extra text\n\n"
        f"Format:\n"
        '{"title": "...", "segments": [{"text": "...", "visual_prompt": "..."}]}'
    )

    def _call():
        response = client.models.generate_content(
            model=SCRIPT_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    return retry(_call, label="script generation")


# ── Step 2: Images ────────────────────────────────────────────────────────────

def generate_image_imagen(visual_prompt: str, out_path: Path):
    """Generate image using Imagen 4 Fast."""
    def _call():
        response = client.models.generate_images(
            model=IMAGE_MODEL,
            prompt=visual_prompt,
            config={"number_of_images": 1, "aspect_ratio": "16:9"},
        )
        image_bytes = response.generated_images[0].image.image_bytes
        Path(out_path).write_bytes(image_bytes)

    retry(_call, label=f"Imagen {out_path.name}")


def generate_image_pollinations(visual_prompt: str, out_path: Path):
    """Generate image via Pollinations.ai free API."""
    safe = urllib.parse.quote(visual_prompt[:200])
    url = f"https://image.pollinations.ai/prompt/{safe}?width=1280&height=720&nologo=true"

    def _call():
        Path(out_path).write_bytes(requests.get(url, timeout=30).content)

    retry(_call, label=f"Pollinations {out_path.name}")


def generate_images(segments: list, out_dir: Path):
    total = len(segments)
    for i, seg in enumerate(segments):
        out_path = out_dir / f"img_{i:04d}.jpg"
        if out_path.exists():
            print(f"  Segment {i+1}/{total} — image already exists, skipping")
            continue

        vp = seg["visual_prompt"]
        if i < IMAGE_QUOTA:
            generate_image_imagen(vp, out_path)
        else:
            generate_image_pollinations(vp, out_path)

        print(f"  Segment {i+1}/{total} — image done")


# ── Step 3: Voiceover (TTS) ───────────────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, out_path: Path, sample_rate: int = 24000):
    """Wrap raw 16-bit mono PCM bytes in a WAV container."""
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def generate_audio(text: str, out_path: Path):
    """Generate a WAV clip for one segment using Gemini TTS."""
    def _call():
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore",
                        )
                    )
                ),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        pcm_to_wav(pcm, out_path)

    retry(_call, label=f"TTS {out_path.name}")


def generate_voiceovers(segments: list, out_dir: Path):
    total = len(segments)
    for i, seg in enumerate(segments):
        out_path = out_dir / f"audio_{i:04d}.wav"
        if out_path.exists():
            print(f"  Segment {i+1}/{total} — audio already exists, skipping")
            continue

        generate_audio(seg["text"], out_path)
        print(f"  Segment {i+1}/{total} — audio done")


# ── Step 4: Assembly via FFmpeg ───────────────────────────────────────────────

def wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def assemble_video(segments: list, out_dir: Path) -> Path:
    total = len(segments)
    clip_paths = []

    print("\n[ASSEMBLY] Building per-segment clips…")
    for i in range(total):
        img_path   = out_dir / f"img_{i:04d}.jpg"
        audio_path = out_dir / f"audio_{i:04d}.wav"
        clip_path  = out_dir / f"clip_{i:04d}.mp4"

        if clip_path.exists():
            print(f"  Clip {i+1}/{total} already exists, skipping")
            clip_paths.append(clip_path)
            continue

        duration = wav_duration(audio_path)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(duration), "-i", str(img_path),
            "-i", str(audio_path),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg clip {i} failed:\n{result.stderr}")

        print(f"  Clip {i+1}/{total} assembled")
        clip_paths.append(clip_path)

    concat_list = out_dir / "concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths))

    final_path = out_dir / "final.mp4"
    print("\n[ASSEMBLY] Concatenating all clips into final.mp4…")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        str(final_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr}")

    print(f"[ASSEMBLY] Done → {final_path}")
    return final_path


# ── Full pipeline for one topic ───────────────────────────────────────────────

def run_pipeline(topic: str):
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    script_cache = OUTPUT_DIR / f"{sanitize(topic)}_script.json"

    if script_cache.exists():
        print("[SCRIPT] Loading cached script…")
        script = json.loads(script_cache.read_text())
    else:
        script = generate_script(topic)
        script_cache.write_text(json.dumps(script, indent=2, ensure_ascii=False))

    title    = sanitize(script["title"])
    segments = script["segments"]
    print(f"[SCRIPT] Title: {script['title']}  |  Segments: {len(segments)}")

    out_dir = OUTPUT_DIR / title
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[IMAGES] Generating {len(segments)} images…")
    generate_images(segments, out_dir)

    print(f"\n[TTS] Generating {len(segments)} audio clips…")
    generate_voiceovers(segments, out_dir)

    final = assemble_video(segments, out_dir)
    print(f"\n✓ Pipeline complete for '{script['title']}'\n  Output: {final}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

TOPICS = [
    "The Hidden History of the Great Wall of China",
    "How Black Holes Actually Work — A Visual Journey",
    "The Rise and Fall of the Roman Empire Explained",
]

if __name__ == "__main__":
    for topic in TOPICS:
        run_pipeline(topic)
