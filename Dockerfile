# Dockerfile

FROM runpod/serverless-hello-world:latest

WORKDIR /app

# 1) Install all Python deps in one go
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy your inference code & adapter bundle
COPY inference.py .
COPY model-inference.tar .

# 3) Extract adapter files into /model
RUN mkdir /model \
 && tar -xzf model-inference.tar -C /model

# 4) Point your code at the model dir & secrets
ENV MODEL_DIR=/model \
    HF_MODEL_ID="huggyllama/llama-7b" \
    HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
    HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_API_TOKEN}

# 5) Launch your inference script
CMD ["python", "-u", "inference.py"]
