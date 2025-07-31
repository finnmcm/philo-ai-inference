FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
    runpod \
    torch==2.0.1 \
    transformers==4.49.0 \
    peft==0.4.0 \
    bitsandbytes==0.39.0 \
    accelerate==0.22.0
COPY handler.py .
COPY model-inference.tar.gz .
RUN mkdir /model && tar -xzf model-inference.tar.gz -C /model
ENV MODEL_DIR=/model HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}
CMD ["python", "-u", "handler.py"]
