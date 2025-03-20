# Use a lightweight Python base image
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /usr/src/app/marker

# Copy only necessary files (use a .dockerignore file for better control)
COPY . .

# Install required system packages and clean up
RUN apt-get update && apt-get install -y \
    build-essential ffmpeg libsm6 libxext6 texlive-extra-utils poppler-utils datamash \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN uv pip install --system --no-cache-dir --upgrade marker-pdf[full] streamlit

# Preprocess the PDF and remove temporary files
RUN marker_single --output_dir /usr/src/app/marker /usr/src/app/marker/test.pdf && \
    rm test.pdf && rm -rf test

# Set the default command to run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.fileWatcherType", "none", "--server.headless", "true"]

# docker build -t szczynk/marker:latest --progress=plain --network=host . && docker run -d -p 11432:8501 --name marker szczynk/marker