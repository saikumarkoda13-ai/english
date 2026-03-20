# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --no-input

# Expose port
EXPOSE 8000

# Run the build script steps (migrate, nltk download)
RUN python -c "import nltk; nltk.download('stopwords')"

# Run gunicorn
CMD ["gunicorn", "Automatic_English_Essay_Scoring_Algorithm_Based_On_Ml.wsgi:application", "--bind", "0.0.0.0:8000"]
