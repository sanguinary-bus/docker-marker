# Use a lightweight Python base image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /usr/src/app/marker

# Copy only necessary files (use a .dockerignore file for better control)
COPY . .

# Install required system packages and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsm6 libxext6 texlive-extra-utils poppler-utils datamash \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -U marker-pdf streamlit

# Preprocess the PDF and remove temporary files
RUN marker_single --output_dir /usr/src/app/marker /usr/src/app/marker/test.pdf && \
    rm test.pdf && rm -rf test

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

