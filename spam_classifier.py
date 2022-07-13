# İstenmeyen E-posta olarak adlandırılan spamların artışı daha güvenilir ve sağlam antispam filtrelerinin geliştirilmesine yönelik yoğun bir ihtiyaç yarattı. 
# Gelen kutumuza düşen promosyon mesajları veya reklamlar, herhangi bir değer sağlamadıkları ve genellikle bizi rahatsız ettikleri için spam olarak sınıflandırılabilir.
# Bu çalışmada UCI'nin Machine Learning Repository'sinden elde edilen veriler kullanılacaktır.
# İlk olarak gerekli paketleri içe aktardık.
"""
import matplotlib.pyplot as plt
import csv
import sklearn
import pickle
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve

""" Colab'da istediğimiz dosyaları **files.upload()** şeklinde tanımlayabiliriz. """

from google.colab import files
uploaded = files.upload()

data = pd.read_csv('spam.csv', encoding='latin-1')
data.head()

data['Label'].value_counts()

import nltk
nltk.download("punkt")
import warnings
warnings.filterwarnings('ignore')

"""Spam kelime bulutu ve ham kelime bulutu oluşturalım."""

ham_words = ''
spam_words = ''

for val in data[data['Label'] == 'spam'].EmailText:
    EmailText = val.lower()
    tokens = nltk.word_tokenize(EmailText)
    for words in tokens:
        spam_words = spam_words + words + ' '

for val in data[data['Label'] == 'ham'].EmailText:
    EmailText = EmailText.lower()
    tokens = nltk.word_tokenize(EmailText)
    for words in tokens:
        ham_words = ham_words + words + ' '

spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)

plt.figure( figsize=(10,8), facecolor='w')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

"""Spam kelime bulutunda **"Free"** ifadesinin spam'de en sık kullanıldığını görebiliriz."""

plt.figure( figsize=(10,8), facecolor='g')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

"""Makinenin anlayabilmesi için ham ve spam ifadelerini 0 ve 1 olarak tanımlayalım."""

data = data.replace(['ham','spam'],[0, 1])
data.head(10)

"""Mesajlardaki noktalama işaretleri ve yasak kelimeleri kaldıralım."""

import nltk
nltk.download('stopwords')

import string
def text_process(text):

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    return " ".join(text)

data['EmailText'] = data['EmailText'].apply(text_process)
data.head()

EmailText = pd.DataFrame(data['EmailText'])
Label = pd.DataFrame(data['Label'])

"""Kelimeleri vektörlere dönüştürelim"""

from collections import Counter

total_counts = Counter()
for i in range(len(EmailText)):
    for word in EmailText.values[i][0].split(" "):
        total_counts[word] += 1

print("Total words in data set: ", len(total_counts))

vocab = sorted(total_counts, key=total_counts.get, reverse=True)
print(vocab[:60])

vocab_size = len(vocab)
word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i

def text_to_vector(text):
    word_vector = np.zeros(vocab_size)
    for word in text.split(" "):
        if word2idx.get(word) is None:
            continue
        else:
            word_vector[word2idx.get(word)] += 1
    return np.array(word_vector)

word_vectors = np.zeros((len(EmailText), len(vocab)), dtype=np.int_)
for i, (_, text_) in enumerate(EmailText.iterrows()):
    word_vectors[i] = text_to_vector(text_[0])

word_vectors.shape

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['EmailText'])
vectors.shape

features = vectors

"""Eğitim ve test setine ayırma"""

X_train, X_test, y_train, y_test = train_test_split(features, data['Label'], test_size=0.15, random_state=111)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier(n_neighbors=49)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=31, random_state=111)

clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

def train(clf, features, targets):
    clf.fit(features, targets)

def predict(clf, features):
    return (clf.predict(features))

pred_scores_word_vectors = []
for k,v in clfs.items():
    train(v, X_train, y_train)
    pred = predict(v, X_test)
    pred_scores_word_vectors.append((k, [accuracy_score(y_test , pred)]))

pred_scores_word_vectors

def find(x):
    if x == 1:
        print ("Message is SPAM")
    else:
        print ("Message is NOT Spam")

"""Bu aşamadan sonra oluşturulacak newtext'e göre **Spam** veya **Spam değil** ifadeleri kullanılabilir."""

newtext = ["Free Entry"]
integers = vectorizer.transform(newtext)

x = mnb.predict(integers)
find(x)

from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred_nb = mnb.predict(X_test)
y_true_nb = y_test
cm = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_nb")
plt.ylabel("y_true_nb")
plt.show()
