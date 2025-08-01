# 1) Base image (public, CPU build / GPU runtime)
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# 2) Install all Python deps in one go
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your inference code & adapter bundle
COPY inference.py .
COPY model-inference.tar .

# 4) Extract adapter files into /model
RUN mkdir /model \
 && tar -xzf model-inference.tar -C /model

# 5) Pre-cache the tokenizer (lightweight)
RUN python - <<EOF
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(
    "huggyllama/llama-7b",
    cache_dir="/model/base",
    use_fast=False
)
EOF

# 6) Point your code at the model dir & secrets
ENV MODEL_DIR=/model \
    HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}

# 7) Launch your inference script
CMD ["python", "-u", "inference.py"]
