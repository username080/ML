from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk import pos_tag,word_tokenize,WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

emails = [
    "Get your free credit offer now",
    "Meeting starts at 2 PM",
    "Pay your credit card debt immediately",
    "Shall we go to the cinema tomorrow?",
    "You won a gift! Click now",
    "I sent the presentation, can you check?",
    "Double your income, start now",
    "Today's meeting is canceled",
    "Register for the money-making system",
    "I need to wake up early tomorrow",
    "Special offer! Free shopping rights",
    "Any plans for the weekend?",
    "Invest now for big profits",
    "Don't forget to submit your homework",
    "Click this link to claim your reward",
    "I won't be at the office today",
    "Here's the easy way to make money",
    "Can you call me around 5?",
    "Today only discount offer!",
    "There might be a server problem",
    "Don't miss free shipping opportunity",
    "The final report is attached",
    "Pay now, interest will be waived!",
    "When is the project presentation?",
    "Congratulations! You won the lottery",
    "Can we postpone tomorrow's meeting?",
    "Guaranteed profit investment opportunity",
    "Create your new email password",
    "This offer is just for you!",
    "Did we have lunch today?",
    "Click to increase your income",
    "Are you taking notes for today's meeting?",
    "Click to earn free Bitcoin",
    "I updated the presentation, please check",
    "Pay your debt now, interest will be removed",
    "Any plans for tonight?",
    "This campaign is only for tonight",
    "I'll leave early tomorrow morning",
    "Get rich easily, start now",
    "Did you open the PDF file?",
    "Evaluate the credit opportunity now",
    "Shall we go out during lunch?",
    "Investing has never been this easy",
    "I shared the project file",
    "Click to get your gift card",
    "Are you joining the call at 3 PM?",
    "Big discounts are waiting for you",
    "I sent the updated list",
    "Here is the secret to making money online",
    "Want to win a free phone?"
]

labels = [
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0
]  

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



def preprocess(sentence):
    sentence.lower()
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    filtered_sentence = []
    for word,tag in tagged:
        if word in stop_words:
            continue
        else:
            filtered_sentence.append(lemmatizer.lemmatize(word.lower(),get_wordnet_pos(tag)))
    
    return (" ".join(filtered_sentence))


filtered_emails = []

for e in emails:
    filtered_emails.append(preprocess(e))


model = MultinomialNB()
tfidf = TfidfVectorizer()

x = tfidf.fit_transform(filtered_emails)
y = labels

model.fit(x,y)

text = "Congratulations u won the competitions"

x_test = tfidf.transform([preprocess(text)])

pred = model.predict(x_test)
print("Spam" if pred[0] == 1 else "Not Spam")
