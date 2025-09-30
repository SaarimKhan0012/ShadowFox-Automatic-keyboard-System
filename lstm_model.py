# lstm_model.py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_lstm_model(corpus, seq_len=3, vocab_size=1000):
    # Tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(corpus)

    # Prepare sequences
    sequences = []
    for line in corpus:
        seq = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(seq)):
            start = max(0, i - seq_len)
            sequences.append(seq[start:i+1])

    # Pad sequences
    sequences = pad_sequences(sequences, maxlen=seq_len+1, padding='pre')
    X = sequences[:, :-1]
    y = sequences[:, -1]

    # Build model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_len))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, tokenizer, X, y
