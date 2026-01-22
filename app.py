# app.py — Streamlit app for Bi-directional RNN/LSTM/GRU sentiment model

import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIG ----------------
MODEL_PATH = "best_sentiment_model.h5"   # change if using BiGRU / BiRNN
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 40
# ----------------------------------------

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("Sentiment Analysis (Bi-directional LSTM)")

# load tokenizer
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

# load model
@st.cache_resource
def load_sentiment_model():
    return load_model(MODEL_PATH)

tokenizer = load_tokenizer()
model = load_sentiment_model()

# input
text = st.text_area("Enter text", height=120)

# predict
if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Enter some text")
    else:
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        score = model.predict(pad)[0][0]

        # confidence-based decision
        if score >= 0.7:
            label = "Positive"
            color = "green"
        elif score <= 0.3:
            label = "Negative"
            color = "red"
        else:
            label = "Uncertain"
            color = "orange"

        st.markdown(f"### Prediction: :{color}[{label}]")
        st.write(f"Confidence score: **{score:.3f}**")

# footer
st.markdown("---")
st.caption("Bi-directional LSTM • Tokenizer + Padding • Binary Sentiment")
