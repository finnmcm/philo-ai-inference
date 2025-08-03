import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import runpod
from runpod.serverless import start

# ——— Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Load quantized model ———
MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
bnb_cfg   = BitsAndBytesConfig(load_in_4bit=True)

logger.info(f"Loading tokenizer from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

logger.info(f"Loading 8-bit model from {MODEL_DIR}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_cfg,
    device_map="auto"
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model ready on {device}")

# ——— Handler ———
def handler(event):
    try:
        payload = json.loads(event["body"]) if isinstance(event.get("body"), str) else event
        prompt  = payload.get("inputs") or payload.get("prompt")
        if not prompt:
            return {"statusCode": 400, "body": json.dumps({"error": "No 'inputs' provided"})}

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.95)

        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {"statusCode": 200, "body": json.dumps({"generated_text": text})}

    except Exception as e:
        logger.exception("Inference error")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

# ——— Start serverless loop ———
if __name__ == "__main__":
    start({"handler": handler})
