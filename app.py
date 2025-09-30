import streamlit as st
from trigram_model import TrigramModel
from sample_corpus import corpus
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# -----------------------------
# Load pretrained LSTM
# -----------------------------
lstm_model = load_model("lstm_nextword.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

seq_len = 3

def lstm_predict(text, top_k=5):
    words = text.lower().split()
    seq = tokenizer.texts_to_sequences([' '.join(words)])[-1]
    seq = pad_sequences([seq[-seq_len:]], maxlen=seq_len, padding='pre')
    preds = lstm_model.predict(seq, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    return [tokenizer.index_word.get(idx, "<UNK>") for idx in top_indices]

# -----------------------------
# Train Trigram Model
# -----------------------------
trigram_model = TrigramModel()
trigram_model.train(corpus)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔮 Live Hybrid Autocorrect Keyboard")
st.write("Start typing; suggestions appear automatically:")

# Initialize session state
if 'text' not in st.session_state:
    st.session_state.text = ""

# Update session state when typing
def update_text():
    st.session_state.text = st.session_state.input_box

st.text_input(
    "Your text:",
    value=st.session_state.text,
    key='input_box',
    on_change=update_text
)

user_input = st.session_state.text

# -----------------------------
# Live Suggestions
# -----------------------------
if user_input.strip() != "":
    words = user_input.split()
    
    # Get predictions from Trigram + LSTM
    trigram_preds = trigram_model.predict(words, top_k=5)
    lstm_preds = lstm_predict(user_input, top_k=5)

    # Merge, remove duplicates, shuffle for variety
    hybrid_preds = list(dict.fromkeys(trigram_preds + lstm_preds))
    random.shuffle(hybrid_preds)
    hybrid_preds = hybrid_preds[:5]  # max 5 suggestions

    # Display suggestions horizontally
    cols = st.columns(len(hybrid_preds))
    for i, w in enumerate(hybrid_preds):
        button_key = f"sugg_{i}_{w}"
        if cols[i].button(w, key=button_key):
            st.session_state.text += " " + w
