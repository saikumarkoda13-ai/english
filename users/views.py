# LAZY LOADING VERSION 3.2 - MEMORY OPTIMIZED
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import re

# Global variable for model caching
_MODEL_CACHE = {}


# =========================
# USER REGISTRATION
# =========================
def UserRegisterActions(request):
    from .forms import UserRegistrationForm
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Exists')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


# =========================
# LOGIN
# =========================
def UserLoginCheck(request):
    from .models import UserRegistrationModel
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid,
                password=pswd
            )
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                return render(request, 'users/UserHomePage.html')
            else:
                messages.success(request, 'Your Account Not Activated')
        except:
            messages.success(request, 'Invalid Login ID and Password')
    return render(request, 'UserLogin.html')


# =========================
# USER HOME
# =========================
def UserHome(request):

    return render(request, 'users/UserHomePage.html')


# =========================
# DATASET VIEW
# =========================
def DatasetView(request):
    import pandas as pd
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT, "training_set_rel3.tsv")
    df = pd.read_csv(
        path,
        sep='\t',
        encoding='ISO-8859-1'
    )

    df.dropna(axis=1, inplace=True)

    # Convert to HTML to avoid ambiguous truth value error in template
    # We limit to 100 rows to ensure the browser doesn't freeze with large datasets
    data_html = df.head(100).to_html(index=False) if not df.empty else None

    return render(
        request,
        'users/viewdataset.html',
        {'data': data_html}
    )


# =========================
# TRAINING
# =========================
def training(request):
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import pandas as pd
    import numpy as np
    from nltk.corpus import stopwords
    from gensim.models import Word2Vec
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from django.conf import settings
    path_tsv = os.path.join(settings.MEDIA_ROOT, "training_set_rel3.tsv")
    df = pd.read_csv(
        path_tsv,
        sep='\t',
        encoding='ISO-8859-1'
    )

    df.dropna(axis=1, inplace=True)

    path_processed = os.path.join(settings.MEDIA_ROOT, "Processed_data.csv")
    temp = pd.read_csv(path_processed)
    temp.drop(columns=["Unnamed: 0"], inplace=True)

    df['domain1_score'] = temp['final_score']

    y = df['domain1_score']
    X = df['essay']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    stop_words = set(stopwords.words('english'))

    def clean(text):

        text = re.sub("[^A-Za-z]", " ", text)
        words = text.lower().split()

        return [w for w in words if w not in stop_words]

    train_words = [clean(e) for e in X_train]
    test_words = [clean(e) for e in X_test]

    # =========================
    # WORD2VEC
    # =========================

    word2vec_model = Word2Vec(
        train_words,
        vector_size=300,
        window=10,
        min_count=40,
        workers=4
    )

    word2vec_model.wv.save_word2vec_format(
        "word2vecmodel.bin",
        binary=True
    )

    def makeVec(words, model):

        vec = np.zeros((300,), dtype="float32")
        count = 0

        for w in words:

            if w in model.wv:
                vec += model.wv[w]
                count += 1

        if count != 0:
            vec /= count

        return vec

    train_vec = np.array([makeVec(w, word2vec_model) for w in train_words])
    test_vec = np.array([makeVec(w, word2vec_model) for w in test_words])

    train_vec = train_vec.reshape(train_vec.shape[0], 1, 300)
    test_vec = test_vec.reshape(test_vec.shape[0], 1, 300)

    # =========================
    # LSTM MODEL
    # =========================

    model = Sequential()

    model.add(LSTM(300, input_shape=(1, 300), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(
        loss='mean_squared_error',
        optimizer='rmsprop',
        metrics=['mae']
    )

    model.fit(
        train_vec,
        y_train,
        batch_size=64,
        epochs=5
    )

    model.save("final_lstm.h5")

    preds = model.predict(test_vec)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    return render(
        request,
        "users/ml.html",
        {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4)
        }
    )


# =========================
# LAZY LOADING CACHE HELPERS
# =========================
def get_ai_models():
    """Helper to lazily load and cache models only when needed."""
    global _MODEL_CACHE
    import os
    
    if "word2vec" not in _MODEL_CACHE:
        from gensim.models import KeyedVectors
        from django.conf import settings
        path = os.path.join(settings.BASE_DIR, "word2vecmodel.bin")
        _MODEL_CACHE["word2vec"] = KeyedVectors.load_word2vec_format(
            path,
            binary=True
        )
        
    if "lstm" not in _MODEL_CACHE:
        from django.conf import settings
        os.environ["KERAS_BACKEND"] = "tensorflow"
        from keras.models import load_model
        path = os.path.join(settings.BASE_DIR, "final_lstm.h5")
        _MODEL_CACHE["lstm"] = load_model(path, safe_mode=False)
        
    return _MODEL_CACHE["word2vec"], _MODEL_CACHE["lstm"]


# =========================
# PREDICTION
# =========================
def prediction(request):

    score = None

    if request.method == "POST":
        try:
            import numpy as np
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            from PIL import Image
            import pytesseract
            
            # Load models lazily
            word2vec_model, lstm_model = get_ai_models()

            final_text = request.POST.get("final_text")
            image_file = request.FILES.get("essay_image")

            # OCR
            if image_file:
                img = Image.open(image_file)
                final_text = pytesseract.image_to_string(img)

            if not final_text:
                return render(
                    request,
                    "users/predictForm.html",
                    {"score": "Please enter essay text"}
                )

            stop_words = set(stopwords.words("english"))

            text = re.sub("[^A-Za-z]", " ", final_text)
            words = text.lower().split()
            words = [w for w in words if w not in stop_words]

            vec = np.zeros((300,), dtype="float32")
            count = 0

            for w in words:
                if w in word2vec_model.key_to_index:
                    vec += word2vec_model[w]
                    count += 1

            if count == 0:
                score = "No valid words found"
            else:
                vec /= count
                vec = vec.reshape(1, 1, 300)
                pred = lstm_model.predict(vec)
                score = str(round(float(pred[0][0])))

            return render(
                request,
                "users/predictForm.html",
                {"score": score}
            )
        except Exception as e:
            # If any exception occurs, catch it and display it on the page instead of 500 error!
            return render(
                request,
                "users/predictForm.html",
                {"score": f"Server Error Caught: {str(e)}"}
            )

    return render(request, "users/predictForm.html")