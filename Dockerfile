# Use Runpod’s public Serverless Python base image
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# 1) Install Python dependencies, including SentencePiece
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentencepiece

# 2) Copy your inference code & adapter bundle
COPY inference.py .
COPY model-inference.tar.gz .

# 3) Extract adapter files into /model
RUN mkdir /model \
 && tar -xzf model-inference.tar.gz -C /model

# 4) Pre-download & cache the base model & tokenizer (now with SentencePiece available)
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
# cache_dir="/model/base"
AutoTokenizer.from_pretrained(
    "huggyllama/llama-7b",
    cache_dir="/model/base",
    use_fast=False
)
AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    cache_dir="/model/base",
    load_in_8bit=True,
    device_map="auto"
)
EOF

# 5) Point inference.py at the right directories & secrets
ENV MODEL_DIR=/model \
    HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}

# 6) Launch your inference script
CMD ["python", "-u", "inference.py"]
