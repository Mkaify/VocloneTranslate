
# 🎙️ VoCloneTranslate

### **Automated End-to-End Audio-Visual Speech Translation & Few-Shot Urdu Voice Cloning Pipeline**

Developed as a Final Year Project (FYP) within the Department of Software Engineering at the **University of Engineering and Technology, Taxila**. 

VoCloneTranslate redefines automated localized media dubbing by replacing flat, robotic text-to-speech outputs with an expressive, chained neural pipeline that dynamically preserves individual speaker characteristics across language boundaries while matching video frame visemes.

---

## 🚀 Core Architectural Pipeline & Workflow

The platform processes input video media streams through five distinct asynchronous processing blocks:

1. **Audio Extraction Subsystem (FFmpeg):** Asynchronously demuxes raw video containers to isolate high-fidelity audio streams.
2. **Optimized Automatic Speech Recognition (OpenAI Whisper):** Decodes source English frames to map raw text strings alongside exact word-level time boundaries.
3. **Neural Machine Translation Service (Meta NLLB-200):** Resolves English-to-Urdu grammatical inversion (SVO to SOV syntax alignment) using specialized language target tokens.
4. **Few-Shot Voice Cloning Engine (XTTS-v2):** Quantizes and applies acoustic features extracted from a brief source reference audio clip to condition expressive Urdu speech generation.
5. **Generative Visual Synchronization Head (Wav2Lip):** Morphs local face frame boundaries frame-by-frame to achieve precise phonetic mouth matching, eliminating structural dubbing lag entirely.
---

## 📂 Repository Architecture

```text
├── app.py                 # Primary FastAPI Gateway Router & Interface Endpoints
├── base_en_model.py       # Core Translation Orchestration & Structural Configs
├── xtts_worker.py         # Asynchronous Execution Hub & Subprocess Pipeline Workers
├── requirements.txt       # Environment Dependency Matrix Blueprint
└── .gitignore             # Strict Local Cache and Heavy Checkpoint Exclusion Mapping
```

## 📥 Project Assets & Model Checkpoints

Due to git file boundary thresholds (100 MB ceiling), the large production model weights and specialized validation datasets are tracked externally via dedicated deep learning hosting channels:

● **Production Model Checkpoints:** ➔ [Download Fine-Tuned XTTS-v2 Model Weights](https://huggingface.co/kashiilambar/vct-xtts-v2-weights) 
● **Validation Media Dataset:** ➔ [Access Customized Urdu Speech Datasets Repository](https://huggingface.co/datasets/kashiilambar/vct-custom_Dataset)

## 📊 Empirical Performance & Resource Optimization

Engineered specifically to execute within tightly constrained hardware infrastructure envelopes:

● **Memory Constraints:** Implemented targeted model quantization layers (`int8_float16` for ASR / `FP16` for translation weights), restricting the maximum operational footprint down to a clean **~5.3 GB VRAM**. This allows the complete pipeline to run on a single standard NVIDIA T4 GPU. ● **Execution Latency Profile (60-Second Video Clip Benchmark):**

-   _Audio Extraction (FFmpeg):_ **0.8s**
    
-   _Speech Recognition (OpenAI Whisper):_ **11.2s**
    
-   _Neural Machine Translation (NLLB-200):_ **5.4s**
    
-   _Voice Cloning Synthesis (XTTS-v2):_ **10.1s**
    
-   _Audio-Visual Rendering (FFmpeg Multiplexer):_ **1.2s**
    
-   **Total Core Processing Time:** **28.7 Seconds**
    

## 🛠️ Local Environment Deployment

1.  **Clone the Codebase & Initialize Dependencies:**
    
    ```bash
    git clone [https://github.com/Mkaify/VocloneTranslate.git](https://github.com/Mkaify/VocloneTranslate.git)
    cd VocloneTranslate
    pip install -r requirements.txt
    
    ```
    
2.  **Download Asset Weights:** Ensure the downloaded model files from Hugging Face are placed into the local directory parameters defined in your application environment file (`.env`).
    
3.  **Launch Production API Service:**
    
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    

## 👥 Engineering & Evaluation Credentials

● **Institution:** University of Engineering and Technology, Taxila (UET Taxila) 
● **Department:** Software Engineering Department 
● **Project Identifiers:** Registration No: 22-SE-02 (Muhammad Kaif ur Rehman) & Project Evaluation Team 
● **Project Supervisor:** Engr. Dr. Marriam Nawaz 
● **Final Evaluation Schedule:** July 2, 2026
