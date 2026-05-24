import os
import shutil
import subprocess
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS

print("🚀 Loading Base Model for Sanity Check...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. LOAD THE GUARANTEED WORKING BASE MODEL
# This bypasses your local model_files entirely to test the AWS machine.
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print("✅ Base Engine Ready on AWS GPU!")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/dub")
async def process_dubbing(audio_file: UploadFile = File(...)):
    temp_input_path = f"/tmp/{audio_file.filename}"
    clean_ref_path = "/tmp/clean_ref.wav"
    output_audio_path = "/tmp/final_test_dub.wav"

    # Save upload
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Clean Audio
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_input_path, 
        "-ac", "1", "-ar", "24000", clean_ref_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Synthesizing Test Audio...")
    
    # 2. GENERATE AUDIO
    # We are using English for this test to ensure the base model doesn't hit any character errors.
    tts.tts_to_file(
        text="Hello, this is a clean test to verify the server is producing normal audio without static.",
        speaker_wav=clean_ref_path,
        language="en",
        file_path=output_audio_path
    )
    
    # Cleanup
    if os.path.exists(temp_input_path): os.remove(temp_input_path)
    if os.path.exists(clean_ref_path): os.remove(clean_ref_path)
    
    return FileResponse(output_audio_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)