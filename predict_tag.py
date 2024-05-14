import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import random

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

# Initialize dataset
dataset = []

# Populate dataset from intents
for intent in intents_data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['responses']
    context_set = intent['context_set']

    # Create an entry for each pattern-response pair
    for pattern in patterns:
        dataset.append({
            'tag': tag,
            'patterns': [pattern],
            'responses': responses,
            'context_set': context_set
        })

# print(dataset)

# nltk.download('stopwords')
# nltk.download('punkt')

# Preprocessing
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()
              and word.lower() not in stop_words]
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

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X, y)

loop = True

print('welcome to the first aid chatbot, how can i help you?')
while loop:
    new_prompt = input()
    if new_prompt == "goodbye":
        print("goodbye")
        loop = False
    else:
        new_prompt_processed = preprocess_text(new_prompt)
        new_prompt_vectorized = vectorizer.transform([new_prompt_processed])
        predicted_tag = classifier.predict(new_prompt_vectorized)[0]
        response = [item['responses']
                    for item in dataset if item.get('tag') == predicted_tag]
        print(f'{random.choice(response)}')
        print('what else can i help you with?')
