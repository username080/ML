from gensim.models import Word2Vec

sentences = [
    ['kahve', 'içtim', 'sabah'],
    ['sabah', 'güneş', 'güzel'],
    ['kahve', 'çok', 'güzel'],
    ['ben', 'sabah', 'koşuya', 'çıktım']
]

model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=1, sg=1)  # sg=0: CBOW, sg=1: Skip-gram

print(model.wv['kahve'])

print(model.wv.most_similar('kahve'))
