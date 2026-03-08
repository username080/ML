import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


text = "The quick brown fox jumps over the lazy dog."



tokens = word_tokenize(text)

tagged_tokens = pos_tag(tokens)

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

f = []

for w,tag in tagged_tokens:
    if w.lower() in ["am","is","are"]:
        f.append(w.lower())
    else:
        if not w.lower() in stop_words:
            f.append(lemmatizer.lemmatize(w.lower(),get_wordnet_pos(tag)))

print(f)