import codecs
import csv
import pickle
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import  KMeans
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
import matplotlib.pyplot as plt

def list_to_str(arr):
    str_ = ''
    for rec in arr:
        str_ += rec
    return str_


def cav_to_list(arr):
    arr_list = []
    for rov in arr:
        arr_list.append(list_to_str(rov))
    return arr_list


def df_preprocess(text):
    reg = re.compile('[^а-яА-яa-zA-Z0-9 ]')
    text = text.lower().replace("ё", "е")
    text = text.replace("ъ", "ь")
    text = text.replace("й", "и")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'сайт', text)
    text = re.sub('@[^\s]+', 'пользователь', text)
    text = reg.sub(' ', text)

    stemmer = SnowballStemmer("russian")
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in russian_stopwords])

    return text


economy = cav_to_list(csv.reader(codecs.open('economy2.csv', 'rU', 'utf-8', errors='ignore')))
cooking = cav_to_list(csv.reader(codecs.open('cooking2.csv', 'rU', 'utf-8', errors='ignore')))
#culture = cav_to_list(csv.reader(codecs.open('culture.csv', 'rU', 'utf-8', errors='ignore')))
sport = cav_to_list(csv.reader(codecs.open('sport2.csv', 'rU', 'utf-8', errors='ignore')))

df_economy = pd.DataFrame(economy, columns=['recall'])
df_economy['type'] = 0
df_economy.head()

df_cooking = pd.DataFrame(cooking, columns=['recall'])
df_cooking['type'] = 1
df_cooking.head()

#df_culture = pd.DataFrame(culture, columns=['recall'])
#df_culture['type'] = 2
#df_culture.head()

df_sport = pd.DataFrame(sport, columns=['recall'])
df_sport['type'] = 3
df_sport.head()

raw_data = pd.concat((df_economy, df_cooking, df_sport), axis=0).sample(frac=1.0)
raw_data.head()

russian_stopwords = stopwords.words("russian")
raw_data['recall'] = raw_data['recall'].apply(df_preprocess)

X = raw_data['recall']
y = raw_data['type']
X_train, X_test, y_train, y_test = train_test_split(raw_data['recall'], raw_data['type'],
                                                    random_state=42,
                                                    test_size=0.3)

russian_stopwords = stopwords.words("russian")

sp_x = []
sp_y = []

for i in range(1,1000,10):
    RandomForest = Pipeline([
        ('vect', CountVectorizer(max_features=1500,
                                 min_df=5,
                                 max_df=0.7,
                                 stop_words=russian_stopwords)),
        ('clf', RandomForestClassifier(random_state=i)),
    ])
    RandomForest.fit(X_train, y_train)
    y_pred = RandomForest.predict(X_test)
    with open('model_classification_RandomForest', 'wb') as picklefile:
        pickle.dump(RandomForest, picklefile)
    print(classification_report(y_test, y_pred))
    print(f"F1 Score случайный лес: {accuracy_score(y_test, y_pred)}")
    sp_x.append(i)
    sp_y.append(accuracy_score(y_test, y_pred))

plt.plot(sp_x,sp_y)
plt.show()