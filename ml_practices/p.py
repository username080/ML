import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier





df = pd.read_csv("comments.csv")
df = shuffle(df, random_state=42).reset_index(drop=True)

labels = df['label']
comments = df['text']

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
model = model = LogisticRegression()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    filtered = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in tagged if w not in stop_words]
    return filtered

filtered_comments = [preprocess(x) for x in comments]

x_train_texts, x_test_texts, y_train, y_test = train_test_split(filtered_comments, labels, train_size=0.7, random_state=42)

w2v_model = Word2Vec(sentences=x_train_texts, vector_size=100, window=5, min_count=1, workers=4)

def get_sentence_vector(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)

x_train_vecs = np.array([get_sentence_vector(tokens) for tokens in x_train_texts])
x_test_vecs  = np.array([get_sentence_vector(tokens) for tokens in x_test_texts])


model.fit(x_train_vecs, y_train)
y_pred = model.predict(x_test_vecs)
print(classification_report(y_test, y_pred))


def predict_sentiment(text):
    tokens = preprocess(text)
    vec = get_sentence_vector(tokens).reshape(1, -1)
    pred = model.predict(vec)
    return "Positive" if pred[0] == 1 else "Negative"


n = "What an awful movie."
print(predict_sentiment(n))
