# Use official Python 3.13 image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (optional, can be needed for pandas/numpy/scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt gunicorn

# Copy the project files into the container
COPY . .

# Expose port 5000 (Flask default, Render maps it to $PORT automatically)
EXPOSE 5000

# Command to run the app with Gunicorn
# Render provides the PORT environment variable dynamically
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app
