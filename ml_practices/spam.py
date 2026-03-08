



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk import pos_tag,word_tokenize,WordNetLemmatizer
import re
from nltk.corpus import stopwords
import pandas as pd

df = pd.read_csv('comments.csv')
emails = df['email'].tolist()
labels = df['label'].tolist()



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


stop_words = set(stopwords.words('english'))
lemmetizer = WordNetLemmatizer()



def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    filtered = [lemmetizer.lemmatize(w,get_wordnet_pos(tag)) for w,tag in tagged if w not in stop_words]
    return (" ".join(filtered))


filtered_emails = []
for e in emails:
    filtered_emails.append(preprocess(e))

x_train, x_test, y_train, y_test = train_test_split(filtered_emails, labels, train_size=0.75, random_state=42)


pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), min_df=2),
    LogisticRegression(max_iter=1000)
)

pipeline.fit(x_train,y_train)
y_pred = pipeline.predict(x_test)

print(classification_report(y_test, y_pred))

n = "Exclusive offer: Let me help you reduce your credit card interest rates. Contact me if interested."
n = preprocess(n)
pred = pipeline.predict([n])
print("Spam" if pred[0] == 1 else "Not Spam")













