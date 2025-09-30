from lstm_model import build_lstm_model
from sample_corpus import corpus
from tensorflow.keras.models import save_model
import pickle

seq_len = 3
vocab_size = 1000  # adjust if needed

# Build LSTM model
model, tokenizer, X, y = build_lstm_model(corpus, seq_len=seq_len, vocab_size=vocab_size)
model.fit(X, y, batch_size=8, epochs=100, verbose=1)

# Save trained model and tokenizer
model.save("lstm_nextword.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ LSTM model trained and saved successfully!")
