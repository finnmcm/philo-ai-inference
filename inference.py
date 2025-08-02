from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os, json, torch, logging, runpod
from runpod.serverless import start

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load local base + LoRA
BASE_PATH = os.environ["BASE_MODEL_PATH"]
LORA_PATH = os.environ["MODEL_DIR"]
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

logger.info(f"Loading tokenizer from {BASE_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, use_fast=False)

logger.info(f"Loading base model from {BASE_PATH}")
base = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

logger.info(f"Applying LoRA adapter from {LORA_PATH}")
model = PeftModel.from_pretrained(base, LORA_PATH, device_map="auto")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def handler(event):
    # … identical to before …
    # parse event["body"] → prompt, generate, return JSON …
    ...

if __name__ == "__main__":
    start({"handler": handler})
