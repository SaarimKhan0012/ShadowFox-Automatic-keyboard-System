# trigram_model.py
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)

    def train(self, corpus):
        for line in corpus:
            words = ["<s>"] + line.lower().split() + ["</s>"]
            for i in range(len(words)):
                self.unigrams[words[i]] += 1
                if i >= 1:
                    self.bigrams[words[i-1]][words[i]] += 1
                if i >= 2:
                    self.trigrams[(words[i-2], words[i-1])][words[i]] += 1

    def predict(self, prev_words, top_k=3):
        candidates = Counter()
        if len(prev_words) >= 2:
            key = (prev_words[-2].lower(), prev_words[-1].lower())
            if key in self.trigrams:
                candidates = self.trigrams[key]
        if not candidates and len(prev_words) >= 1:
            key = prev_words[-1].lower()
            if key in self.bigrams:
                candidates = self.bigrams[key]
        if not candidates:
            candidates = self.unigrams
        return [w for w, _ in candidates.most_common(top_k)]
