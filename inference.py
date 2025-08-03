import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from runpod import RPOD_LOGGER, server

MODEL_DIR = "/model"

# 1) Set up a 4-bit quant config. We choose NF4 + fp16 compute:
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# 2) Load base + quant
RPOD_LOGGER.info(f"Loading base model (4-bit) from {MODEL_DIR}…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

# 3) Attach LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    device_map="auto",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

def handler(request):
    prompt = request["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    server.start({"handler": handler})
