from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Eğitim verisi
emails = [
    "Ücretsiz kredi teklifinizi alın",
    "Toplantı saat 3'te",
    "Kredi kartı borcunuzu hemen ödeyin",
    "Bugün hava çok güzel",
    "Ücretsiz hediye fırsatı sizi bekliyor",
    "Yarın rapor teslimi var"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = normal

vectorizer = CountVectorizer(ngram_range=(1,2))

model = make_pipeline(vectorizer, MultinomialNB())

model.fit(emails, labels)

test_email = ["Ücretsiz kredi fırsatını kaçırmayın"]

pred = model.predict(test_email)
print("Spam" if pred[0] == 1 else "Normal")
