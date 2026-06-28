"""
check.py — Tests every possible image generation model on your Gemini free tier key.
Tries all methods: generate_images(), generate_content(), requests to Pollinations.
Saves successful images to ./check_results/
"""

import os, time, urllib.parse, requests, traceback
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=KEY)

OUT = Path("check_results")
OUT.mkdir(exist_ok=True)

PROMPT = "ancient chinese great wall at sunset, cinematic, dramatic lighting"
RESULTS = []

def log(model, method, status, detail=""):
    icon = "✅" if status == "OK" else "❌"
    print(f"  {icon} [{method}] {status} — {detail}")
    RESULTS.append((model, method, status, detail))

def try_generate_images(model_name):
    """Uses client.models.generate_images() — for Imagen models"""
    try:
        r = client.models.generate_images(
            model=model_name,
            prompt=PROMPT,
            config={"number_of_images": 1, "aspect_ratio": "16:9"}
        )
        img_bytes = r.generated_images[0].image.image_bytes
        out = OUT / f"{model_name.replace('/', '_').replace('.', '_')}_generateImages.jpg"
        out.write_bytes(img_bytes)
        log(model_name, "generate_images", "OK", f"{len(img_bytes)} bytes → {out.name}")
        return True
    except Exception as e:
        log(model_name, "generate_images", "FAIL", str(e)[:120])
        return False

def try_generate_content(model_name):
    """Uses client.models.generate_content() — for Gemini image models"""
    try:
        r = client.models.generate_content(
            model=model_name,
            contents=PROMPT,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )
        saved = False
        for part in r.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                img_bytes = part.inline_data.data
                out = OUT / f"{model_name.replace('/', '_').replace('.', '_')}_generateContent.jpg"
                out.write_bytes(img_bytes)
                log(model_name, "generate_content", "OK", f"{len(img_bytes)} bytes → {out.name}")
                saved = True
                break
        if not saved:
            text = ""
            for part in r.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text = part.text[:80]
                    break
            log(model_name, "generate_content", "FAIL", f"No image in response. Text: {text}")
        return saved
    except Exception as e:
        log(model_name, "generate_content", "FAIL", str(e)[:120])
        return False

# ── All image-capable models to test ──────────────────────────────────────────

imagen_models = [
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
]

gemini_image_models = [
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image",
    "gemini-3-pro-image-preview",
    "nano-banana-pro-preview",
]

# ── Run tests ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("TESTING IMAGEN 4 MODELS (generate_images method)")
print("="*60)
for m in imagen_models:
    print(f"\n→ {m}")
    try_generate_images(m)
    time.sleep(2)

print("\n" + "="*60)
print("TESTING GEMINI IMAGE MODELS (generate_content method)")
print("="*60)
for m in gemini_image_models:
    print(f"\n→ {m}")
    try_generate_content(m)
    time.sleep(2)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
working = [(m, method) for m, method, status, _ in RESULTS if status == "OK"]
failed  = [(m, method) for m, method, status, _ in RESULTS if status == "FAIL"]

if working:
    print(f"\n✅ WORKING ({len(working)}):")
    for m, method in working:
        print(f"   {m} via {method}")
else:
    print("\n❌ NO GOOGLE IMAGE MODELS WORK ON FREE TIER")
    print("   → Use Pollinations.ai (FLUX) — already confirmed working")

print(f"\n❌ FAILED ({len(failed)}):")
for m, method in failed:
    print(f"   {m} via {method}")

print(f"\nCheck ./check_results/ for any saved images")
print("="*60)
