FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and general utilities
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the API requirements
COPY requirements-api.txt .

# Install PyTorch CPU first to leverage Docker layer caching
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the API dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the rest of the backend files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 7860

# Command to run the FastAPI app (Hugging Face Spaces uses port 7860 by default)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
