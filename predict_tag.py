import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#nltk.download('stopwords')
#nltk.download('punkt')

# Sample dataset
dataset = [
    {
        "tag": "Wound Care",
        "patterns": [
            "I cut my finger while chopping vegetables",
            "I scraped my knee on the pavement",
            # Add more patterns as needed
        ],
        "responses": [
            "For minor cuts and scrapes, clean the wound with soap and water, apply an antibiotic ointment, and cover it with a bandage.",
            "Ensure the wound is clean, apply pressure if it's bleeding, and use a bandage to cover it.",
            # Add more responses as needed
        ],
        "context_set": ""
    },
    {
        "tag": "Sore Throat",
        "patterns": [
            "I have a tickle in my throat",
            "It feels painful everytime I swallow",
            # Add more patterns as needed
        ],
        "responses": [
            "You likely have a cold. Try taking an ibuprofen to subside the pain in your throat.",
            "Try drinking tea or warm water with lemon to ease your soreness and help with your cold.",
            # Add more responses as needed
        ],
        "context_set": ""
    },
    # Add more entries for other tags as needed
]

# Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(tokens)

# Extract features using TF-IDF
corpus = []
tags = []
for entry in dataset:
    for pattern in entry['patterns']:
        corpus.append(preprocess_text(pattern))
        tags.append(entry['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Target labels
y = tags

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X, y)

# Example usage: Predicting the tag for a new prompt
new_prompt = "I got stabbed by a knife"
new_prompt_processed = preprocess_text(new_prompt)
new_prompt_vectorized = vectorizer.transform([new_prompt_processed])
predicted_tag = classifier.predict(new_prompt_vectorized)[0]
print("Predicted tag:", predicted_tag)