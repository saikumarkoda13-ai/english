import os
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    BASE_DIR=os.path.dirname(os.path.abspath(__file__)),
    INSTALLED_APPS=[
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'admins',
        'users',
    ],
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': 'db.sqlite3'}},
)
django.setup()

from users.views import get_ai_models
import numpy as np

# Test lazy loading
print("Loading models...")
try:
    wv, lstm = get_ai_models()
    print("Models loaded successfully")
except Exception as e:
    print(f"FAILED TO LOAD MODELS: {e}")

# Simulate prediction text processing
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    import re

    stop_words = set(stopwords.words("english"))
    final_text = "This is a wonderful test essay for the system to evaluate."
    
    text = re.sub("[^A-Za-z]", " ", final_text)
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]

    vec = np.zeros((300,), dtype="float32")
    count = 0
    for w in words:
        if w in wv.key_to_index:
            vec += wv[w]
            count += 1
            
    if count == 0:
        score = "No valid words found"
    else:
        vec /= count
        vec = vec.reshape(1, 1, 300)
        pred = lstm.predict(vec)
        score = str(round(float(pred[0][0])))
    print(f"PREDICTION SCORE is: {score}")

except Exception as e:
    print(f"FAILED PREDICTION LOGIC: {e}")
