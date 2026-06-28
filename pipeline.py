"""
YouTube Video Pipeline
Generates a segmented script, images, voiceover, and assembles a final MP4.
"""

import os
import json
import time
import wave
import struct
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path

import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

SCRIPT_MODEL = "gemini-3.1-flash-lite-latest"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
IMAGEN_MODEL = "imagen-4.0-fast-generate-001"
IMAGEN_QUOTA = 25          # first N segments use Imagen; rest use Pollinations
TARGET_SEGMENTS = 70
WORDS_PER_SEGMENT = 22
MAX_RETRIES = 3

OUTPUT_DIR = Path("output")


# ── Helpers ───────────────────────────────────────────────────────────────────

def retry(fn, *args, attempts=MAX_RETRIES, delay=5, label="call", **kwargs):
    """Call fn(*args, **kwargs) up to `attempts` times, sleeping `delay` s between tries."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  [retry {attempt}/{attempts}] {label} failed: {exc}. Retrying in {delay}s…")
            time.sleep(delay)


def sanitize(name: str) -> str:
    """Make a string safe for use as a directory/file name."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip().replace(" ", "_")


# ── Step 1: Script generation ─────────────────────────────────────────────────

def generate_script(topic: str) -> dict:
    print(f"\n[SCRIPT] Generating script for: {topic}")

    prompt = (
        f"Create a YouTube video script about: {topic}\n\n"
        f"Requirements:\n"
        f"- Exactly {TARGET_SEGMENTS} segments\n"
        f"- Each segment has ~{WORDS_PER_SEGMENT} spoken words (narrator text)\n"
        f"- Each segment has a short, vivid visual_prompt for AI image generation\n"
        f"- Output ONLY valid JSON, no markdown fences, no extra text\n\n"
        f"Format:\n"
        '{"title": "...", "segments": [{"text": "...", "visual_prompt": "..."}]}'
    )

    def _call():
        model = genai.GenerativeModel(SCRIPT_MODEL)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    return retry(_call, label="script generation")


# ── Step 2: Image generation ──────────────────────────────────────────────────

def generate_image_imagen(visual_prompt: str, out_path: Path):
    """Generate image using Imagen 4 Fast."""
    def _call():
        client = genai.ImageGenerationModel(IMAGEN_MODEL)
        result = client.generate_images(
            prompt=visual_prompt,
            number_of_images=1,
            aspect_ratio="16:9",
        )
        img = result.generated_images[0]
        out_path.write_bytes(img.image.image_bytes)

    retry(_call, label=f"Imagen {out_path.name}")


def generate_image_pollinations(visual_prompt: str, out_path: Path):
    """Generate image via Pollinations.ai free API."""
    encoded = urllib.parse.quote(visual_prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&nologo=true"

    def _call():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        out_path.write_bytes(data)

    retry(_call, label=f"Pollinations {out_path.name}")


def generate_images(segments: list, out_dir: Path):
    total = len(segments)
    for i, seg in enumerate(segments):
        out_path = out_dir / f"img_{i:04d}.jpg"
        if out_path.exists():
            print(f"  Segment {i+1}/{total} — image already exists, skipping")
            continue

        vp = seg["visual_prompt"]
        if i < IMAGEN_QUOTA:
            generate_image_imagen(vp, out_path)
        else:
            generate_image_pollinations(vp, out_path)

        print(f"  Segment {i+1}/{total} — image done")


# ── Step 3: Voiceover (TTS) ───────────────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, out_path: Path, sample_rate=24000, channels=1, sampwidth=2):
    """Wrap raw PCM bytes in a WAV container."""
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def generate_audio(text: str, out_path: Path):
    """Generate a WAV clip for one segment using Gemini TTS."""
    def _call():
        client = genai.GenerativeModel(TTS_MODEL)
        response = client.generate_content(
            text,
            generation_config=genai.GenerationConfig(
                response_modalities=["AUDIO"],
                speech_config=genai.SpeechConfig(
                    voice_config=genai.VoiceConfig(
                        prebuilt_voice_config=genai.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                ),
            ),
        )
        # Extract PCM audio from response
        audio_part = response.candidates[0].content.parts[0]
        pcm = audio_part.inline_data.data
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

def get_audio_duration(wav_path: Path) -> float:
    """Return duration in seconds of a WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


def assemble_video(segments: list, out_dir: Path) -> Path:
    total = len(segments)
    clip_paths = []

    print("\n[ASSEMBLY] Building per-segment clips…")
    for i in range(total):
        img_path = out_dir / f"img_{i:04d}.jpg"
        audio_path = out_dir / f"audio_{i:04d}.wav"
        clip_path = out_dir / f"clip_{i:04d}.mp4"

        if clip_path.exists():
            print(f"  Clip {i+1}/{total} already exists, skipping")
            clip_paths.append(clip_path)
            continue

        duration = get_audio_duration(audio_path)
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

    # Write concat list
    concat_list = out_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths)
    )

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

    # ── Script
    script_cache = OUTPUT_DIR / f"{sanitize(topic)}_script.json"
    if script_cache.exists():
        print("[SCRIPT] Loading cached script…")
        script = json.loads(script_cache.read_text())
    else:
        script = generate_script(topic)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        script_cache.write_text(json.dumps(script, indent=2, ensure_ascii=False))

    title = sanitize(script["title"])
    segments = script["segments"]
    print(f"[SCRIPT] Title: {script['title']}  |  Segments: {len(segments)}")

    out_dir = OUTPUT_DIR / title
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Images
    print(f"\n[IMAGES] Generating {len(segments)} images…")
    generate_images(segments, out_dir)

    # ── Voiceovers
    print(f"\n[TTS] Generating {len(segments)} audio clips…")
    generate_voiceovers(segments, out_dir)

    # ── Assembly
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
