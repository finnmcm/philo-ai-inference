# download_base.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

MODEL_ID = "huggyllama/llama-7b"
OUT_DIR  = Path("/tmp/base")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Download & save tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer.save_pretrained(OUT_DIR)

# 2) Download & save 8-bit quantized model
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cpu"
)
model.save_pretrained(OUT_DIR)
