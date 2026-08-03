import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm
from nltk.corpus import treebank

# -----------------------------
# Download Required Datasets
# -----------------------------
print("Checking and downloading required NLTK resources...\n")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('treebank')

# -----------------------------
# Input Tweet
# -----------------------------
tweet = input("Enter a tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\n==============================")
print("TOKENIZATION")
print("==============================")
print(tokens)

# -----------------------------
# N-Gram Language Model
# -----------------------------
print("\n==============================")
print("N-GRAM LANGUAGE MODEL")
print("==============================")

# Unigrams
unigrams = list(ngrams(tokens, 1))
print("\nUnigrams:")
for gram in unigrams:
    print(gram)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
for gram in bigrams:
    print(gram)

# Trigrams
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
for gram in trigrams:
    print(gram)

# Word Frequency Distribution
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(f"{word} : {freq}")

# Most Common Words
print("\nMost Common Words:")
for word, freq in fd.most_common(5):
    print(f"{word} : {freq}")

# -----------------------------
# Hidden Markov Model (HMM)
# -----------------------------
print("\n==============================")
print("HIDDEN MARKOV MODEL (HMM)")
print("==============================")

print("\nTraining HMM POS Tagger... Please wait.")

# Train HMM using Treebank corpus
train_data = treebank.tagged_sents()[:3000]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Predict POS Tags
tagged_sentence = hmm_tagger.tag(tokens)

print("\nPredicted POS Tags:")
for word, tag in tagged_sentence:
    print(f"{word} --> {tag}")

# -----------------------------
# Comparison
# -----------------------------
print("\n==============================")
print("COMPARISON OF N-GRAM AND HMM")
print("==============================")

print("\nN-Gram Model")
print("----------------------------------------")
print("1. Learns sequences of words.")
print("2. Predicts the next word using previous words.")
print("3. Uses word probabilities.")
print("4. Used in text generation, autocomplete, and language modeling.")

print("\nHidden Markov Model (HMM)")
print("----------------------------------------")
print("1. Predicts hidden states (POS tags).")
print("2. Uses transition and emission probabilities.")
print("3. Performs sequence labeling.")
print("4. Used in POS tagging, speech recognition, and NLP tasks.")

print("\n==============================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("==============================")
