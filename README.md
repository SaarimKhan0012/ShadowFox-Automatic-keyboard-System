# ShadowFox-Automatic-keyboard-System

A live predictive keyboard built with Python and Streamlit that suggests the next word as you type. It combines Trigram language modeling and LSTM neural networks to provide accurate, context-aware suggestions. The app can optionally fetch datasets via an API key or use a built-in fallback corpus.

Features

Live word suggestions as you type (no Enter required)

Hybrid prediction: Trigram + LSTM

Clickable suggestions automatically appended to input

Supports small built-in corpus or external API-based datasets

Fast and lightweight for demonstration purposes

Fully built with Python, Streamlit, and TensorFlow/Keras

Folder Structure

autocorrect_keyboard/

├── app.py                # Streamlit app with live suggestions

├── lstm_model.py         # LSTM model builder

├── trigram_model.py      # Trigram prediction model

├── sample_corpus.py      # Corpus loader (API or fallback)

├── train_lstm.py         # Script to train and save LSTM model

├── lstm_nextword.h5      # Saved trained LSTM model

├── tokenizer.pkl         # Saved tokenizer for LSTM

