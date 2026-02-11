# 1. Base image
FROM python:3.11-slim

# 2. Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4a. Copy dependency specification
COPY requirements.txt .

# 5. Install Python dependencies (pin versions in requirements.txt)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 6. Copy your library source code
COPY src/ ./src

# 4b. Copy setup file
COPY setup.py .

# 7. Optional: install your library itself in editable mode for tests
RUN pip install -e .

# 8. Default command for CI
CMD ["pytest", "tests/"]

