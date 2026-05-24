import os
import shutil
import subprocess
import torch
import whisper
import torchaudio
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# --- 1. THE URDU TOKENIZER PATCH ---
# Prevents the base XTTS engine from stripping Urdu characters
def custom_preprocess_text(self, txt, lang):
    if lang == "ur": return txt.strip()
    if hasattr(self, 'original_preprocess'): return self.original_preprocess(txt, lang)
    raise NotImplementedError(f"Language '{lang}' is not supported.")

if not hasattr(VoiceBpeTokenizer, 'original_preprocess'):
    VoiceBpeTokenizer.original_preprocess = VoiceBpeTokenizer.preprocess_text
    VoiceBpeTokenizer.preprocess_text = custom_preprocess_text

# --- 2. INITIALIZE MODELS ---
print("🚀 Initializing VoCloneTranslate Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Whisper STT...")
whisper_model = whisper.load_model("small").to(device)

print("Loading NLLB Translation...")
# Uses NllbTokenizerFast and Safetensors to bypass library bugs and security blocks
tokenizer_nllb = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M", 
    torch_dtype=torch.float16, 
    use_safetensors=True
).to(device)

# --- 3. LOAD XTTS ---
print("Loading XTTS with Fine-Tuned Weights...")
model_dir = "/home/ubuntu/voclonetranslate/model_files"
config = XttsConfig()
config.load_json(os.path.join(model_dir, "config.json"))
config.model_args.vocab_size = 8497 # Custom Urdu vocabulary dimension
xtts_model = Xtts.init_from_config(config)

# Link to local Urdu-supporting vocabulary
xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=os.path.join(model_dir, "vocab.json"))
xtts_model.tokenizer.char_limits["ur"] = 250

# Weight loading with layer-matching for stability
checkpoint = torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
state_dict = checkpoint.get("model", checkpoint)
model_dict = xtts_model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
xtts_model.load_state_dict(model_dict)
xtts_model.to(device).eval()

print("✅ Engine Ready on AWS GPU!")

# --- 4. FASTAPI SERVER ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/dub")
async def process_dubbing(audio_file: UploadFile = File(...)):
    temp_input_path = f"/tmp/{audio_file.filename}"
    clean_ref_path = "/tmp/clean_ref.wav"
    output_audio_path = "/tmp/final_dub.wav"

    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Clean and Resample to 24kHz for XTTS compatibility
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_input_path, 
        "-ac", "1", "-ar", "24000", clean_ref_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Process: STT -> Translation
    print("1. Transcribing...")
    english_text = whisper_model.transcribe(temp_input_path)["text"].strip()
    
    print("2. Translating...")
    inputs = tokenizer_nllb(english_text, return_tensors="pt").to(device)
    translated_tokens = model_nllb.generate(
        **inputs, 
        forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids("urd_Arab")
    )
    urdu_text = tokenizer_nllb.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"   [Urdu]: {urdu_text}")

    # Synthesis with Stability Tuning for 4.24 Loss Model
    print("3. Synthesizing Voice...")
    with torch.no_grad():
        outputs = xtts_model.synthesize(
            urdu_text, 
            config, 
            speaker_wav=clean_ref_path, 
            language="ur",
            temperature=0.6,          # Balanced for speech vs stability
            repetition_penalty=2.0,
            gpt_cond_len=15,
            top_p=0.8,
            enable_text_splitting=True
        )

    # --- 5. AUDIO NORMALIZATION & SAVING ---
    # Fixes the 'quiet/unplayable' file issue
    wav_data = np.array(outputs['wav'])
    max_val = np.max(np.abs(wav_data))
    
    if max_val > 0:
        # Boost volume to a professional 0.95 peak
        wav_data = wav_data / max_val * 0.95
        print(f"   [Audio] Peak Energy: {max_val:.4f} -> Normalized to 0.95")
    else:
        print("   ⚠️ WARNING: Synthesis returned silence.")

    # Proper tensor formatting for torchaudio
    wav_tensor = torch.tensor(wav_data).float().unsqueeze(0)
    torchaudio.save(output_audio_path, wav_tensor, 24000)
    
    # Final cleanup of temp uploads
    if os.path.exists(temp_input_path): os.remove(temp_input_path)
    if os.path.exists(clean_ref_path): os.remove(clean_ref_path)
    
    return FileResponse(output_audio_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)