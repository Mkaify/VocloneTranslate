import os
import shutil
import subprocess
import torch
import whisper
import torchaudio
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from huggingface_hub import hf_hub_download

# --- 1. THE URDU TOKENIZER PATCH ---
def custom_preprocess_text(self, txt, lang):
    if lang == "ur": return txt.strip()
    if hasattr(self, 'original_preprocess'): return self.original_preprocess(txt, lang)
    raise NotImplementedError(f"Language '{lang}' is not supported.")

if not hasattr(VoiceBpeTokenizer, 'original_preprocess'):
    VoiceBpeTokenizer.original_preprocess = VoiceBpeTokenizer.preprocess_text
    VoiceBpeTokenizer.preprocess_text = custom_preprocess_text

# --- 2. INITIALIZE STT & TRANSLATION MODELS ---
print("Loading Models into VRAM... Please wait.")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Whisper...")
whisper_model = whisper.load_model("small").to(device)

print("Loading NLLB Translation...")
tokenizer_nllb = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
model_nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16, use_safetensors=True).to(device)

# --- 3. LOAD FINE-TUNED XTTS ---
print("Loading XTTS Model...")
model_dir = "/home/ubuntu/voclonetranslate/model_files"
config_path = os.path.join(model_dir, "config.json")

config = XttsConfig()
config.load_json(config_path)
config.model_args.vocab_size = 8497  # Lock to native Urdu patch dimension
xtts_model = Xtts.init_from_config(config)

# print("Fetching pristine official vocabulary...")
print("Fetching VoCloneTranslate Custom Vocabulary..")
# pristine_vocab_path = hf_hub_download(repo_id="coqui/XTTS-v2", filename="vocab.json")
vocab_path = os.path.join(model_dir, "vocab.json")
# xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=pristine_vocab_path)
xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
xtts_model.tokenizer.char_limits["ur"] = 250

print("Loading VoCloneTranslate fine-tuned weights...")
checkpoint_path = os.path.join(model_dir, "model.pth")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Extract the state dict
state_dict = checkpoint.get("model", checkpoint)

# The FIX: Filter the state dict to ensure we are only loading layers
# that exist in initialized model to avoid "dimention mismatch" or silent layer ignoring
model_dict = xtts_model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

model_dict.update(pretrained_dict)
xtts_model.load_state_dict(model_dict)

# The GPT CONDITIONING Fix
xtts_model.gpt.cond_v = torch.nn.Parameter(torch.zeros(1, 1024).to(device))
xtts_model.cuda().eval()

print("✅ Engine Ready on AWS GPU!")

# --- 4. FASTAPI SERVER ---
app = FastAPI(title="VoCloneTranslate API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/dub")
async def process_dubbing(audio_file: UploadFile = File(...)):
    # File paths
    temp_input_path = f"/tmp/upload_{audio_file.filename}"
    clean_ref_path = "/tmp/clean_reference.wav"
    output_audio_path = "/tmp/final_urdu_dub.wav"

    # Save uploaded file (could be video or audio)
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    print("1. Extracting/Cleaning Audio for XTTS...")
    # THE FIX: Force conversion to 22050Hz Mono WAV to prevent silent NaN tensors
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_input_path, 
        "-ac", "1", "-ar", "22050", clean_ref_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("2. Transcribing English...")
    english_text = whisper_model.transcribe(temp_input_path)["text"].strip()
    print(f"   [English]: {english_text}")

    print("3. Translating to Urdu...")
    inputs = tokenizer_nllb(english_text, return_tensors="pt").to(device)
    translated_tokens = model_nllb.generate(**inputs, forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids("urd_Arab"))
    urdu_text = tokenizer_nllb.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"   [Urdu]: {urdu_text}")

    print("4. Synthesizing Urdu Audio...")
    inf_params = {
        "temperature": 0.85,
        "length_penalty": 1.0,
        "repetition_penalty": 2.0,
        "top_k": 50,
        "top_p": 0.80,
        "gpt_cond_len": 30,
        "gpt_cond_chunk_len": 4,
        "max_ref_len": 10,
        "sound_norm_refs": False
    }

    for attr, value in inf_params.items():
        if not hasattr(config, attr):
            setattr(config, attr, value)

    # Use the cleaned WAV file, NOT the raw upload
    with torch.no_grad():
        outputs = xtts_model.synthesize(urdu_text, config, speaker_wav=clean_ref_path, language="ur", **inf_params)

    torchaudio.save(output_audio_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)
    
    # Cleanup temps
    os.remove(temp_input_path)
    os.remove(clean_ref_path)
    
    return FileResponse(output_audio_path, media_type="audio/wav", filename="urdu_dub.wav")

if __name__ == "__main__":
    # Binds to AWS Public IP instantly
    uvicorn.run(app, host="0.0.0.0", port=8000)
