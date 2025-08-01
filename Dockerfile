# Use Runpod’s public Serverless Python base image
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# 1) Install Python dependencies (now using bitsandbytes-cuda117==0.26.0.post2)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy your inference code & adapter bundle
COPY inference.py .
COPY model-inference.tar .

# 3) Extract adapter files into /model
RUN mkdir /model \
 && tar -xzf model-inference.tar -C /model

# 4) Pre-download & cache the base model into /model/base
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# 5) Tell your code where to find the adapter and base model
ENV MODEL_DIR=/model

# 6) Launch your inference script
CMD ["python", "-u", "inference.py"]
