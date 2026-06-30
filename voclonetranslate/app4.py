import os
import torch
import torchaudio
import subprocess
import whisper
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from deep_translator import GoogleTranslator
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import sys
import transformers.pytorch_utils

# Force-inject the missing utility into the system's active module dictionary
transformers.pytorch_utils.isin_mps_friendly = torch.isin
sys.modules['transformers.pytorch_utils'].isin_mps_friendly = torch.isin

app = FastAPI(title="VoCloneTranslate Automated Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/uploads"
OUTPUT_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/outputs"
CHECKPOINT_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/training_pipeline/ft_output/GPT_XTTS_FT-May-16-2026_06+47PM-0000000"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Launching Production Server Layout Sequential Initialization...")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("● Loading Whisper Speech-To-Text Core...")
stt_engine = whisper.load_model("small").to(device)

print("● Loading XTTS Architecture Layout...")
config = XttsConfig()
config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
model.to(device)

print("✅ All Model Pipelines Bound and Ready on AWS GPU Device Vector!")

@app.post("/dub")
async def process_dubbing(audio_file: UploadFile = File(...)):
    input_file_path = os.path.join(UPLOAD_DIR, audio_file.filename)
    extracted_audio_path = os.path.join(UPLOAD_DIR, f"extracted_{os.path.splitext(audio_file.filename)[0]}.wav")
    final_output_path = os.path.join(OUTPUT_DIR, "final_dub.wav")

    try:
        with open(input_file_path, "wb") as storage_buffer:
            storage_buffer.write(await audio_file.read())

        subprocess.run([
            "ffmpeg", "-y", "-i", input_file_path,
            "-ac", "1", "-ar", "24000", extracted_audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 1. English Speech Transcription
        transcription_result = stt_engine.transcribe(extracted_audio_path)
        source_english_text = transcription_result["text"].strip()
        print(f"   [Source Text]: {source_english_text}")

        if not source_english_text:
            raise HTTPException(status_code=400, detail="Transcription parsing pass returned an empty array.")

        # 2. Complete Paragraph Translation
        synthesized_urdu_text = GoogleTranslator(source='en', target='ur').translate(source_english_text)
        print(f"   [Urdu Text]: {synthesized_urdu_text}")

        # 3. Dynamic Paragraph Segmentation Loop
        # Automatically parses any paragraph length down into model-compliant syntax fragments
        punctuation_marks = ["۔", "؟", "!", "،", ".", "?", " - "]
        urdu_text_segments = [synthesized_urdu_text]
        
        for mark in punctuation_marks:
            temporary_pool = []
            for item in urdu_text_segments:
                temporary_pool.extend([part.strip() for part in item.split(mark) if part.strip()])
            urdu_text_segments = temporary_pool

        # Filter out trailing punctuation ghost fragments
        clean_text_segments = [seg for seg in urdu_text_segments if len(seg.strip()) > 1]
        print(f"   [Processing Queue]: Dispatched {len(clean_text_segments)} text segments sequentially...")

        generated_waveform_segments = []

        # 4. Sequential Inference Execution Matrix
        for index, text_chunk in enumerate(clean_text_segments):
            # Force close sentences to protect alignment matrices
            if not text_chunk.endswith(("۔", "؟", "!", ".")):
                text_chunk += " ۔"
                
            print(f"     -> Dubbing Chunk [{index + 1}/{len(clean_text_segments)}]: {text_chunk}")
            
            with torch.no_grad():
                outputs = model.synthesize(
                    text_chunk,
                    config,
                    speaker_wav=extracted_audio_path,
                    language="ur"
                )
            
            # Extract raw audio data array directly
            chunk_tensor = torch.tensor(outputs["wav"]).flatten().float()
            if chunk_tensor.numel() > 0:
                generated_waveform_segments.append(chunk_tensor)

        if not generated_waveform_segments:
            raise HTTPException(status_code=500, detail="The synthesis layer failed to assemble valid audio buffers.")

        # 5. High-Fidelity Waveform Stitching and Audio Production
        print("⚡ Merging structural waveforms down to high-fidelity master sample...")
        final_waveform_flat = torch.cat(generated_waveform_segments, dim=0)
        final_waveform = final_waveform_flat.unsqueeze(0) 
        
        torchaudio.save(final_output_path, final_waveform, 24000)
        print(f"💾 Audio saved successfully. Total duration: {final_waveform.shape[1]/24000:.2f} seconds.")

        # Storage maintenance clean
        if os.path.exists(extracted_audio_path): os.remove(extracted_audio_path)
        if os.path.exists(input_file_path): os.remove(input_file_path)

        return FileResponse(final_output_path, media_type="audio/wav", filename="dubbed_media.wav")

    except Exception as e:
        print(f"❌ Pipeline Operational Failure inside Chunk Layer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching Production Web Gateway on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)