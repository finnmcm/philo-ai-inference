FROM python:3.11-slim
WORKDIR /app

# 1. Copy & install dependencies (adding awscli)
COPY requirements.txt .
RUN pip install --no-cache-dir \
      runpod \
      torch==2.0.1 \
      transformers==4.49.0 \
      bitsandbytes==0.37.0 \
      sentencepiece>=0.1.99 \
      awscli \
      accelerate>=0.22.0 \
      peft>=0.4.0

# 2. Copy your code + startup script
COPY inference.py download_and_run.sh /app/
RUN chmod +x /app/download_and_run.sh

# 3. Tell the container where to find your S3 model
#    (Override via runpod.yaml or `docker run -e MODEL_S3_PATH=…`.)
ENV MODEL_S3_PATH=s3://philo-ai/models/llama8bit/pytorch-training-2025-08-03-03-41-44-353/output/model.tar.gz
ENV AWS_DEFAULT_REGION=us-east-1

# 4. Use the script as the entrypoint
ENTRYPOINT ["/app/download_and_run.sh"]
