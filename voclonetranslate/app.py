import os
import torch
import torchaudio
import subprocess
import whisper
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from deep_translator import GoogleTranslator
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI(title="VoCloneTranslate Automated Production API")

# Configure CORS for local cloud communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/home/ubuntu/voclonetranslate/uploads"
OUTPUT_DIR = "/home/ubuntu/voclonetranslate/outputs"
CHECKPOINT_DIR = "/home/ubuntu/voclonetranslate/training_pipeline/ft_output/GPT_XTTS_FT-May-16-2026_06+47PM-0000000"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SAFE STARTUP WRAPPER: Loading models sequentially to prevent Linux OOM crashes
try:
    print("🧠 [1/2] Loading OpenAI Whisper STT Engine...")
    stt_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 [2/2] Loading Voice Synthesis Cluster (XTTS)...")
    config = XttsConfig()
    config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
    model.cuda()
    print("✅ All AI clusters successfully locked into GPU memory.")
except Exception as e:
    print(f"❌ Critical Startup Failure: Memory allocation failed. Details: {str(e)}")
    exit(1)

# Tokenizer Patch for Genuine Urdu Support
tokenizer = model.tokenizer
for attr in ['langs', 'supported_languages', 'languages']:
    if hasattr(tokenizer, attr):
        current_langs = getattr(tokenizer, attr)
        if "ur" not in current_langs: current_langs.append("ur")

original_preprocess = tokenizer.preprocess_text
tokenizer.preprocess_text = lambda text, lang: text.strip() if lang == "ur" else original_preprocess(text, lang)

@app.post("/api/production-dub")
async def production_dub(file: UploadFile = File(...)):
    input_file_path = os.path.join(UPLOAD_DIR, file.filename)
    extracted_audio_path = os.path.join(UPLOAD_DIR, f"extracted_{file.filename.split('.')[0]}.wav")
    final_output_path = os.path.join(OUTPUT_DIR, f"dubbed_{file.filename.split('.')[0]}.wav")
    
    try:
        # Step 1: Ingest User Media
        with open(input_file_path, "wb") as f:
            f.write(await file.read())
            
        # Step 2: FFmpeg Normalization Matrix
        print(f"🎬 Processing file metadata for: {file.filename}")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_file_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            extracted_audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 3: Automated Speech-to-Text (Whisper Inference)
        print("📝 Running Automated Speech Recognition...")
        stt_result = stt_model.transcribe(extracted_audio_path, language="en")
        english_text = stt_result["text"]
        print(f"🎯 Recognized Text (English): '{english_text}'")
        
        if not english_text.strip():
            raise HTTPException(status_code=400, detail="Audio track contains zero audible English speech signals.")

        # Step 4: Text Translation Neural Matrix
        print("🌐 Converting linguistic structures to Urdu...")
        urdu_text = GoogleTranslator(source='en', target='ur').translate(english_text)
        urdu_text = urdu_text.strip() + " . "
        print(f"✅ Target Translation (Urdu): '{urdu_text}'")
        
        # Step 5: Target Identity Voice Synthesis
        print("🎙️ Initiating Zero-Shot Voice Cloning into Urdu...")
        outputs = model.synthesize(
            urdu_text,
            config,
            speaker_wav=extracted_audio_path, # Zero-shot voice cloning: Uses the user's voice to dub the output!
            gpt_cond_len=3,
            language="ur",
            temperature=0.65,
            length_penalty=1.0,
            repetition_penalty=2.0,
            top_k=50,
            top_p=0.8
        )
        
        # Step 6: Finalize File Export
        torchaudio.save(final_output_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
        
        # Cleanup temporary files to save disk volume
        if os.path.exists(extracted_audio_path): os.remove(extracted_audio_path)
        if os.path.exists(input_file_path): os.remove(input_file_path)
            
        return FileResponse(final_output_path, media_type="audio/wav", filename="dubbed_media.wav")
        
    except Exception as e:
        print(f"❌ Pipeline Operational Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching Production Web Gateway on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
