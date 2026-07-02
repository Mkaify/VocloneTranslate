import os
import soundfile as sf
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME = "suhaibrashid17/UAT_Data"
SAMPLE_COUNT = 500  # Increased for better phonetic coverage
OUTPUT_DIR = "dataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

# Ensure directories exist
os.makedirs(WAV_DIR, exist_ok=True)

print(f"🚀 Downloading {SAMPLE_COUNT} samples from {DATASET_NAME}...")
# Note: Ensure you have downgraded 'datasets' as discussed to avoid torchcodec errors
dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

metadata = []
print("Extracting audio and formatting metadata...")

for i, row in enumerate(tqdm(dataset, total=SAMPLE_COUNT)):
    if i >= SAMPLE_COUNT:
        break

    # 1. Extract Audio & Text Data
    audio_data = row["audio"]["array"]
    sample_rate = row["audio"]["sampling_rate"]
    text = row["text"]

    # 2. Define File Path for saving
    wav_filename = f"urdu_sample_{i:04d}.wav"
    wav_filepath = os.path.join(WAV_DIR, wav_filename)

    # 3. Save the actual audio file
    sf.write(wav_filepath, audio_data, sample_rate)

    # 4. Format for Coqui XTTS (LJSPEECH Formatter)
    # THE CRITICAL FIX: The formatter expects ONLY the ID, no 'wavs/' and no '.wav'
    metadata.append({
        "audio_file": f"urdu_sample_{i:04d}",
        "text": text.strip(),
        "speaker_name": "uat_speaker"
    })

# 5. Save the metadata.csv with the pipe (|) separator
metadata_df = pd.DataFrame(metadata)
csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
metadata_df.to_csv(csv_path, sep="|", index=False, header=False)

print(f"\n✅ Data Prep Complete!")
print(f"   - Saved {SAMPLE_COUNT} audio files to: {WAV_DIR}")
print(f"   - Saved cleaned metadata file to: {csv_path}")