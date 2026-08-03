import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# -----------------------------
# Download Required NLTK Data
# -----------------------------
nltk.download('punkt')

# -----------------------------
# Input Customer Reviews
# -----------------------------
reviews = []

n = int(input("Enter the number of customer reviews: "))

for i in range(n):
    review = input(f"Enter Review {i+1}: ")
    reviews.append(review)

# -----------------------------
# Convert Text into Bag of Words
# -----------------------------
vectorizer = CountVectorizer(stop_words='english')

X = vectorizer.fit_transform(reviews)

# -----------------------------
# LDA Topic Modeling
# -----------------------------
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\n================================")
print("DISCOVERED TOPICS")
print("================================")

for i, topic in enumerate(lda.components_):

    print(f"\nTopic {i+1}")

    top_words = topic.argsort()[-5:]

    for index in reversed(top_words):
        print(words[index])

# -----------------------------
# t-SNE Visualization
# -----------------------------
X_dense = X.toarray()

# Adjust perplexity automatically
perplexity_value = min(2, len(reviews)-1)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity_value
)

X_tsne = tsne.fit_transform(X_dense)

print("\n================================")
print("t-SNE COORDINATES")
print("================================")

for i, point in enumerate(X_tsne):
    print(f"Review {i+1}: {point}")

# -----------------------------
# Plot Reviews
# -----------------------------
plt.figure(figsize=(8,6))

plt.scatter(X_tsne[:,0], X_tsne[:,1], s=100)

for i in range(len(reviews)):
    plt.text(
        X_tsne[i,0],
        X_tsne[i,1],
        "R"+str(i+1),
        fontsize=10
    )

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)

plt.show()

print("\nProgram Executed Successfully.")
