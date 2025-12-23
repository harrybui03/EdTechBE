# Sử dụng Python 3.11 slim image để giảm kích thước
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cài đặt ffmpeg để extract audio từ HLS
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker cache layer
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Tạo non-root user để chạy ứng dụng
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Chown files cho appuser
RUN chown -R appuser:appuser /app

# Chuyển sang non-root user
USER appuser

# Expose port (nếu cần, nhưng worker này không expose port)
# EXPOSE 8080

# Run application
CMD ["python", "main.py"]

