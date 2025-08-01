# 1) Base image (public, GPU‐enabled runtime)
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# 2) Install Python dependencies (with scipy, numpy<2, GPU bitsandbytes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy inference code & adapter bundle
COPY inference.py .
COPY model-inference.tar.gz .

# 4) Extract adapter files
RUN mkdir /model && \
    tar -xzf model-inference.tar.gz -C /model

# 5) Precache base model & tokenizer in 8-bit
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

# 6) Point inference.py at the right dirs & secrets
ENV MODEL_DIR=/model \
    HF_HOME=/model/base \
    HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}

# 7) Launch
CMD ["python", "-u", "inference.py"]
