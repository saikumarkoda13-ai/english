#!/bin/bash

# Run database migrations
echo "Running migrations..."
python manage.py migrate --no-input

# Run gunicorn
echo "Starting Gunicorn..."
gunicorn Automatic_English_Essay_Scoring_Algorithm_Based_On_Ml.wsgi:application --bind 0.0.0.0:8000
