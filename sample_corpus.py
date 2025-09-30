import requests
import re

# -----------------------------
# API Details
# -----------------------------
API_KEY = "YOUR_API_KEY_HERE"
API_URL = "https://your-dataset-api.com/text"  # replace with your API endpoint

# -----------------------------
# Fetch data from API
# -----------------------------
headers = {"Authorization": f"Bearer {API_KEY}"}
try:
    response = requests.get(API_URL, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    print("Error fetching dataset from API:", e)
    data = {"sentences": []}

# -----------------------------
# Extract sentences
# -----------------------------
raw_sentences = data.get("sentences", [])

# -----------------------------
# Preprocess sentences
# -----------------------------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

corpus = [clean_text(sentence) for sentence in raw_sentences if sentence.strip()]

# Fallback small corpus if API fails
if not corpus:
    corpus = [
        "i love machine learning",
        "deep learning is powerful",
        "coding is fun and exciting",
        "artificial intelligence is the future",
        "python is my favorite programming language",
        "i enjoy solving problems",
        "learning new things is amazing",
        "today is a sunny day",
        "the quick brown fox jumps over the lazy dog",
        "she enjoys reading books"
    ]
