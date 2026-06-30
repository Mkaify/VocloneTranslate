import os
import gc
import types
import torch
from huggingface_hub import hf_hub_download
from trainer import Trainer, TrainerArgs
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

MODEL_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/training_pipeline/ft_output/GPT_XTTS_FT-May-12-2026_05+39PM-0000000"
DATASET_DIR = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/training_pipeline/dataset/custom_voice_data"
OUT_PATH = "/home/ubuntu/VocloneTranslate_MVP/voclonetranslate/training_pipeline/ft_output"
os.makedirs(OUT_PATH, exist_ok=True)

print("Locating missing base files...")
dvae_path = hf_hub_download(repo_id="coqui/XTTS-v2", filename="dvae.pth")
mel_stats_path = hf_hub_download(repo_id="coqui/XTTS-v2", filename="mel_stats.pth")

print("Loading configuration...")
config = GPTTrainerConfig()
config.load_json(os.path.join(MODEL_DIR, "config.json"))

config.model_args.tokenizer_file = os.path.join(MODEL_DIR, "vocab.json")
config.model_args.xtts_checkpoint = os.path.join(MODEL_DIR, "model.pth")
config.model_args.dvae_checkpoint = dvae_path
config.model_args.mel_norm_file = mel_stats_path

# --- THE 5 EPOCH UPGRADE ---
config.epochs = 12
config.batch_size = 2
config.eval_batch_size = 2
config.save_step = 50
config.print_step = 10
config.run_eval = True
config.num_loader_workers = 0 

# Low learning rate to safely resume from the 4.24 state
config.lr = 2e-6

if hasattr(config, 'optimizer_params') and 'lr' in config.optimizer_params:
    del config.optimizer_params['lr']

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path=DATASET_DIR,
    language="ur"
)
config.datasets = [dataset_config]

print("Initializing GPTTrainer...")
model = GPTTrainer.init_from_config(config)

print("Applying Urdu Tokenizer Patch...")
model.xtts.tokenizer.char_limits["ur"] = 250
original_preprocess = model.xtts.tokenizer.preprocess_text

def custom_preprocess_text(self, txt, lang):
    if lang == "ur": return txt.strip()
    return original_preprocess(txt, lang)

model.xtts.tokenizer.preprocess_text = types.MethodType(custom_preprocess_text, model.xtts.tokenizer)

# Freeze non-GPT params
for name, param in model.named_parameters():
    if "gpt" not in name:
        param.requires_grad = False

import types
# --- FIX: Unwrap Optimizer List for PyTorch Scheduler Compatibility ---
original_get_scheduler = model.get_scheduler

def safe_get_scheduler(self, optimizer):
    if isinstance(optimizer, list):
        optimizer = optimizer[0]
    return original_get_scheduler(optimizer)

model.get_scheduler = types.MethodType(safe_get_scheduler, model)
# ----------------------------------------------------------------------

print("Loading 300 dataset samples...")
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.1)

trainer_args = TrainerArgs(
    restore_path=None,
    skip_train_epoch=False,
    start_with_eval=True, 
    grad_accum_steps=4
)

trainer = Trainer(
    trainer_args, config, OUT_PATH,
    model=model, train_samples=train_samples, eval_samples=eval_samples
)

print("🚀 Starting 5-Epoch Fine-Tuning Run...")
trainer.fit()
