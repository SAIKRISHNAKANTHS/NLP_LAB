import nltk
from nltk.corpus import brown
from nltk.tag import hmm
from nltk.tokenize import word_tokenize

# Download resources
nltk.download('brown')
nltk.download('punkt')

# Load Brown corpus with original Brown tags
train_data = brown.tagged_sents()

# Train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Input
text = input("Enter a sentence: ")

# Tokenize
tokens = word_tokenize(text)

# POS Tagging
tagged_words = hmm_tagger.tag(tokens)

# Display Tokens
print("\nTokens:")
print(tokens)

# Display POS Tags
print("\nPOS Tags:")
for word, tag in tagged_words:
    print(word, "->", tag)

# Tag meanings
print("\nTag Meanings:")
print("NN  -> Noun")
print("VB  -> Verb")
print("JJ  -> Adjective")
print("RB  -> Adverb")
print("PRP -> Pronoun")
print("DT  -> Determiner")

print("\nTotal Words:", len(tokens))