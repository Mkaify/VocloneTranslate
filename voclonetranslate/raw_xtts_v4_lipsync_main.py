import os
import json
import torch
import torchaudio
import subprocess
import whisper
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from deep_translator import GoogleTranslator
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI(title="VoCloneTranslate Automated Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcription", "X-Translation"]
)

UPLOAD_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/uploads"
OUTPUT_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/outputs"
TEMP_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/temp_chunks"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing VoCloneTranslate Production Pipeline on {device}...")

print("● Loading Whisper Speech-To-Text Core...")
# Using word/segment timestamps requires loading the model natively
stt_engine = whisper.load_model("small").to(device)

print("● Downloading & Loading XTTS-v2-Urdu-FT Core...")
xtts_model_dir = snapshot_download(repo_id="suhaibrashid17/XTTS-v2-Urdu-FT")
xtts_config = XttsConfig()
xtts_config.load_json(os.path.join(xtts_model_dir, "config.json"))

tts_model = Xtts.init_from_config(xtts_config)
tts_model.load_checkpoint(xtts_config, checkpoint_dir=xtts_model_dir, eval=True)
tts_model.to(device)

print("✅ All Model Pipelines Bound and Ready!")


def run_lip_sync(video_chunk_path: str, audio_chunk_path: str, output_chunk_path: str):
    """
    Executes lip-syncing logic to morph speaker lip layouts to the generated Urdu speech track.
    If your Wav2Lip setup is a standalone repository or sub-module, call it via subprocess here.
    """
    try:
        # Fallback tracking if Wav2Lip processing is skipped/not loaded:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_chunk_path, "-i", audio_chunk_path,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            output_chunk_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Lip sync execution exception: {e}")
        shutil.copy(video_chunk_path, output_chunk_path)


@app.post("/dub")
async def process_dubbing(audio_file: UploadFile = File(...)):
    is_video = audio_file.content_type.startswith("video/")
    safe_ext = os.path.splitext(audio_file.filename)[1] or (".mp4" if is_video else ".wav")
    base_filename = os.path.splitext(audio_file.filename)[0].replace(" ", "_")
    
    input_file_path      = os.path.join(UPLOAD_DIR, f"src_{base_filename}{safe_ext}")
    extracted_audio_path = os.path.join(UPLOAD_DIR, f"extracted_{base_filename}.wav")
    final_output_path    = os.path.join(OUTPUT_DIR, f"dubbed_{base_filename}{safe_ext}")

    # Clean runtime workspace directory for slice sequencing
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        with open(input_file_path, "wb") as storage_buffer:
            storage_buffer.write(await audio_file.read())

        print("● Extracting base audio stream layer...")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file_path,
            "-ac", "1", "-ar", "24000", extracted_audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("● Parsing speech segment timelines via Whisper...")
        transcription_result = stt_engine.transcribe(extracted_audio_path, task="transcribe", language="en")
        segments = transcription_result.get("segments", [])

        if not segments:
            raise HTTPException(status_code=400, detail="Transcription passing phase failed to resolve timelines.")

        full_english_text_list = []
        full_urdu_text_list = []
        video_chunk_files = []

        # 2. Process Segment Iterations Natively
        for idx, segment in enumerate(segments):
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            text_chunk = segment["text"].strip()

            if not text_chunk or duration < 0.3:
                continue

            full_english_text_list.append(text_chunk)
            print(f"   -> Processing Segment [{idx+1}]: ({start_time:.2f}s - {end_time:.2f}s) | {text_chunk}")

            urdu_text = GoogleTranslator(source='en', target='ur').translate(text_chunk)
            full_urdu_text_list.append(urdu_text)

            src_video_chunk = os.path.join(TEMP_DIR, f"v_src_{idx}.mp4")
            raw_audio_chunk = os.path.join(TEMP_DIR, f"a_raw_{idx}.wav")
            speed_audio_chunk = os.path.join(TEMP_DIR, f"a_speed_{idx}.wav")
            out_sync_chunk = os.path.join(TEMP_DIR, f"out_sync_{idx}.mp4" if is_video else f"out_sync_{idx}.wav")

            if is_video:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time),
                    "-i", input_file_path, "-c:v", "copy", "-an", src_video_chunk
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time),
                    "-i", input_file_path, "-vn", "-acodec", "copy", src_video_chunk
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            try:
                outputs = tts_model.synthesize(
                    urdu_text, xtts_config, speaker_wav=extracted_audio_path, gpt_cond_len=3, language="ur"
                )
                waveform = torch.tensor(outputs["wav"]).unsqueeze(0) if torch.tensor(outputs["wav"]).dim() == 1 else torch.tensor(outputs["wav"])
                torchaudio.save(raw_audio_chunk, waveform, 24000)
                
                # ── FIXED: Calculate actual length directly from the tensor shape without hitting the disk ──
                generated_duration = waveform.shape[-1] / 24000.0

            except Exception as e:
                print(f"      ⚠️ Waveform generation failed on slice {idx}, applying silent space buffer.")
                subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:c=1", "-t", str(duration), raw_audio_chunk], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                generated_duration = duration

            # Calculate required time-stretch ratio factor to match original video duration
            stretch_ratio = generated_duration / duration
            stretch_ratio = max(0.6, min(1.8, stretch_ratio))

            # Apply FFmpeg audio tempo filter stretching modifications
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_audio_chunk,
                "-filter:a", f"atempo={stretch_ratio}",
                speed_audio_chunk
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if is_video:
                run_lip_sync(src_video_chunk, speed_audio_chunk, out_sync_chunk)
            else:
                shutil.copy(speed_audio_chunk, out_sync_chunk)

            video_chunk_files.append(out_sync_chunk)

        if not video_chunk_files:
            raise HTTPException(status_code=500, detail="The synchronization array pipeline processed zero tracks.")

        concat_list_path = os.path.join(TEMP_DIR, "structure_map.txt")
        with open(concat_list_path, "w") as map_file:
            for file_path in video_chunk_files:
                map_file.write(f"file '{file_path}'\n")

        print("● Reassembling timeline fragments into cohesive output stream...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy" if not is_video else "h264", "-c:a", "aac", final_output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(extracted_audio_path): os.remove(extracted_audio_path)
        if os.path.exists(input_file_path): os.remove(input_file_path)
        shutil.rmtree(TEMP_DIR)

        headers = {
            "X-Transcription": json.dumps(" ".join(full_english_text_list)),
            "X-Translation":   json.dumps(" ".join(full_urdu_text_list)),
        }

        return FileResponse(
            final_output_path,
            media_type="video/mp4" if is_video else "audio/wav",
            filename=f"dubbed_lecture{safe_ext}",
            headers=headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Pipeline Architectural Crash: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
