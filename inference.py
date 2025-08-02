import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import runpod
from runpod.serverless import start

# ——————— CONFIGURE LOGGING ———————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——————— COLD START: LOAD TOKENIZER & MODEL ———————
MODEL_DIR   = os.environ.get("MODEL_DIR", "/model")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "huggyllama/llama-7b")

logger.info(f"Loading tokenizer from '{HF_MODEL_ID}'…")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)

logger.info(f"Loading base model '{HF_MODEL_ID}' (8-bit) on device_map='auto'…")
base_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    load_in_8bit=True,
    device_map="auto"
)

logger.info(f"Applying LoRA adapter from '{MODEL_DIR}'…")
model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    device_map="auto"
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model loaded and ready on {device}")

# ——————— DEFAULT GENERATION PARAMETERS ———————
GEN_KWARGS = {
    "max_new_tokens": 128,
    "temperature":   0.7,
    "top_p":         0.95,
    "do_sample":     True,
}

# ——————— HANDLER ———————
def handler(event):
    """
    event: dict parsed from the incoming JSON body.
    Expects either event["body"] (a JSON string) or event["inputs"] directly.
    """
    try:
        # 1) Parse JSON body
        payload = (
            json.loads(event["body"])
            if isinstance(event.get("body"), str)
            else event
        )
        prompt = payload.get("inputs") or payload.get("prompt")
        if not prompt:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No 'inputs' field provided."})
            }

        # 2) Tokenize + generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **GEN_KWARGS)

        # 3) Decode and return
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "statusCode": 200,
            "body": json.dumps({"generated_text": generated})
        }

    except Exception as e:
        logger.exception("Error in handler")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# ——————— START THE SERVERLESS LOOP ———————
if __name__ == "__main__":
    # This will bind the above `handler` to your /run route
    start({"handler": handler})
