import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download tokenizer
nltk.download('punkt')

# Keywords representing biomedical relations
keywords = ["treats", "reduces", "controls", "helps"]

# User input
sentence = input("Enter biomedical sentence: ")
actual = int(input("Actual Relation (1/0): "))

# Tokenize sentence
tokens = word_tokenize(sentence.lower())

print("\nTokens:")
print(tokens)

# Predict relation
predicted = 0

for word in tokens:
    if word in keywords:
        predicted = 1

print("\nPredicted Relation:", predicted)

# Evaluation
y_true = [actual]
y_pred = [predicted]

precision = precision_score(y_true, y_pred, zero_division=0)#tp/tp+fp
recall = recall_score(y_true, y_pred, zero_division=0)#tp/tp+fn
f1 = f1_score(y_true, y_pred, zero_division=0)#2*(p*r)/(p+r)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
