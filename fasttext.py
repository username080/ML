from gensim.models import FastText

# Örnek cümleler (tokenize edilmiş)
sentences = [
    ['kahve', 'içtim', 'sabah'],
    ['sabah', 'güneş', 'güzel'],
    ['kahve', 'çok', 'güzel'],
    ['ben', 'sabah', 'koşuya', 'çıktım']
]

model = FastText(sentences, vector_size=50, window=3, min_count=1, workers=1, sg=1)  # sg=1: skip-gram

print(model.wv['kahve'])

print(model.wv.most_similar('kahve'))

print(model.wv['kahveler'])
