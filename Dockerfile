FROM runpod/serverless-hello-world:latest
WORKDIR /app

# 1) Install & pin deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y triton  || echo "no external triton to uninstall"

# 2) Copy your code + adapter
COPY inference.py model-inference.tar .

# 3) Unpack your LoRA adapter
RUN mkdir /model \
 && tar -xzf model-inference.tar -C /model

# 4) Env for your script
ENV MODEL_DIR=/model \
    HF_MODEL_ID=huggyllama/llama-7b \
    HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
    HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_API_TOKEN}

# 5) Launch
CMD ["python", "-u", "inference.py"]
