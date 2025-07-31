FROM ghcr.io/runpod/python-serverless:latest

# Copy your code & model tarball
WORKDIR /app
COPY inference.py .
COPY model-inference.tar.gz .

# Install your dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Extract your adapter files
RUN mkdir /model && \
    tar -xzf model-inference.tar.gz -C /model

# Tell Runpod which function to call
ENV MODEL_DIR=/model
