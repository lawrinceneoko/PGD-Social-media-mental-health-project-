"""
train_models_api.py

Trains:
 - TF-IDF vectorizer + ANN (dense) classifier on TF-IDF features
 - Tokenizer + LSTM classifier on sequences

Saves:
 - tfidf_vectorizer.joblib
 - ann_model.h5
 - tokenizer.joblib
 - lstm_model.h5
 - model_comparison.csv

Usage:
    python train_models_api.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silence TF logs
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CONFIG
DATA_CSV = "mental_dataset.csv"
TFIDF_FILE = "tfidf_vectorizer.joblib"
ANN_MODEL_FILE = "ann_model.h5"
TOKENIZER_FILE = "tokenizer.joblib"
LSTM_MODEL_FILE = "lstm_model.h5"
COMPARISON_FILE = "model_comparison.csv"

MAX_WORDS = 8000
MAX_LEN = 150
EMBED_DIM = 64
ANN_EPOCHS = 5
LSTM_EPOCHS = 4
BATCH_SIZE = 64

def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower()
    t = re.sub(r'(https?://\S+|www\.\S+)', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def load_data(csv_path=DATA_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run dataset_api_loader.py first.")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(int)
    return df

def train_tfidf_ann(X_texts, y):
    print("Fitting TF-IDF...")
    vect = TfidfVectorizer(max_features=5000)
    X_tfidf = vect.fit_transform(X_texts)
    joblib.dump(vect, TFIDF_FILE)
    print("Saved TF-IDF vectorizer ->", TFIDF_FILE)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    input_dim = X_train.shape[1]

    print("Building ANN (dense) model on TF-IDF features...")
    ann = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ann.fit(X_train.toarray(), y_train, validation_split=0.1, epochs=ANN_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    ann.save(ANN_MODEL_FILE)
    print("Saved ANN model ->", ANN_MODEL_FILE)

    y_pred = (ann.predict(X_test.toarray()) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    print(f"ANN: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
    return {"system": "ANN (TF-IDF)", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def train_lstm(X_texts, y):
    print("Preparing tokenizer for LSTM...")
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(X_texts)
    joblib.dump(tok, TOKENIZER_FILE)
    print("Saved tokenizer ->", TOKENIZER_FILE)
    sequences = tok.texts_to_sequences(X_texts)
    X_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42, stratify=y)

    print("Building LSTM model...")
    lstm = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm.fit(X_train, y_train, validation_split=0.1, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    lstm.save(LSTM_MODEL_FILE)
    print("Saved LSTM model ->", LSTM_MODEL_FILE)

    y_pred = (lstm.predict(X_test) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    print(f"LSTM: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
    return {"system": "LSTM", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    df = load_data()
    X_texts = df["text"].tolist()
    y = df["label"].values

    ann_metrics = train_tfidf_ann(X_texts, y)
    lstm_metrics = train_lstm(X_texts, y)

    # create comparison CSV (include a simple existing baseline placeholder)
    baseline = {"system": "existing_baseline", "accuracy": 0.60, "precision": 0.60, "recall": 0.60, "f1": 0.60}
    comp_df = pd.DataFrame([baseline, ann_metrics, lstm_metrics])
    comp_df.to_csv(COMPARISON_FILE, index=False)
    print(f"Saved comparison metrics -> {COMPARISON_FILE}")
    print(comp_df)

if __name__ == "__main__":
    main()
