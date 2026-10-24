"""
app_mental_api.py

Minimal Flask app that loads the saved TF-IDF+ANN and Tokenizer+LSTM models and
offers a tiny web UI to paste social-media posts (one per line) and get predictions.

Usage:
    python app_mental_api.py
Open http://127.0.0.1:5000/
"""
import os
import re
from flask import Flask, render_template, request
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Config -- filenames saved by training script
TFIDF_FILE = "tfidf_vectorizer.joblib"
ANN_MODEL_FILE = "ann_model.h5"
TOKENIZER_FILE = "tokenizer.joblib"
LSTM_MODEL_FILE = "lstm_model.h5"

MAX_LEN = 150
LSTM_WEIGHT = 0.6
ANN_WEIGHT = 0.4
THRESHOLD = 0.5

app = Flask(__name__)

# ---------------------------
# Load models and vectorizers
# ---------------------------
def clean_text_simple(t):
    if t is None:
        return ""
    s = str(t).lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

print("Loading vectorizer and models... (make sure you ran train_models_api.py)")
if not os.path.exists(TFIDF_FILE) or not os.path.exists(ANN_MODEL_FILE) or not os.path.exists(TOKENIZER_FILE) or not os.path.exists(LSTM_MODEL_FILE):
    raise FileNotFoundError("One or more model files are missing. Run train_models_api.py first.")

vectorizer = joblib.load(TFIDF_FILE)
ann_model = load_model(ANN_MODEL_FILE)
tokenizer = joblib.load(TOKENIZER_FILE)
lstm_model = load_model(LSTM_MODEL_FILE)

# ---------------------------
# Prediction helper
# ---------------------------
def predict_texts(texts):
    cleaned = [clean_text_simple(t) for t in texts]
    # ANN predictions (TF-IDF)
    X_tfidf = vectorizer.transform(cleaned)
    ann_probs = ann_model.predict(X_tfidf.toarray()).flatten()
    # LSTM predictions
    seqs = tokenizer.texts_to_sequences(cleaned)
    X_seq = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    lstm_probs = lstm_model.predict(X_seq).flatten()
    # Weighted ensemble
    probs = LSTM_WEIGHT * lstm_probs + ANN_WEIGHT * ann_probs
    preds = (probs >= THRESHOLD).astype(int)
    results = []
    for t, p, pr in zip(texts, probs, preds):
        results.append({"text": t, "prediction": int(pr), "confidence": float(round(float(p), 4))})
    return results

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    messages = ""
    if request.method == "POST":
        messages = request.form.get("messages", "").strip()
        lines = [l.strip() for l in messages.split("\n") if l.strip()]
        if lines:
            results = predict_texts(lines)
    return render_template("index_mental.html", results=results, messages=messages)

if __name__ == "__main__":
    app.run(debug=True)
