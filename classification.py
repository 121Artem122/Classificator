import pandas as pd
import codecs
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import csv
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle


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
sport = cav_to_list(csv.reader(codecs.open('sport2.csv', 'rU', 'utf-8', errors='ignore')))

df_economy = pd.DataFrame(economy, columns=['recall'])
df_economy['type'] = 0
df_economy.head()

df_cooking = pd.DataFrame(cooking, columns=['recall'])
df_cooking['type'] = 1
df_cooking.head()

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

# -----------------------------------------------------------------------------
# Случайный лес
RandomForest = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators=7,
                                   max_depth=8,
                                   min_samples_split=18,
                                   max_features= 'sqrt')),
])
result = RandomForest.fit(X_train, y_train)
y_pred = RandomForest.predict(X_test)
with open('model_classification_RandomForest', 'wb') as picklefile:
    pickle.dump(RandomForest, picklefile)
print(classification_report(y_test, y_pred))
print(f"F1 Score случайный лес: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------------------------------------------------
# Логистическая регрессия
LogReg = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(n_jobs=3, C=1e5, solver='saga',
                               multi_class='multinomial',
                               max_iter=200,
                               random_state=42)),
])
LogReg.fit(X_train, y_train)
y_pred1 = LogReg.predict(X_test)
with open('model_classification_LogReg', 'wb') as picklefile:
    pickle.dump(LogReg, picklefile)
print(classification_report(y_test, y_pred1))
print(f"F1 Score логистическая регрессия: {accuracy_score(y_test, y_pred1)}")
print(confusion_matrix(y_test, y_pred1))

# -----------------------------------------------------------------------------
# k-ближайших соседей
KNN = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('knn', KNeighborsClassifier(n_neighbors=33,
                                 algorithm = 'brute',
                                 p = 2)),
     ])
KNN.fit(X_train, y_train)
y_pred2 = KNN.predict(X_test)
with open('model_classification_KNN', 'wb') as picklefile:
    pickle.dump(KNN, picklefile)
print(classification_report(y_test, y_pred2))
print(f"F1 Score лk-ближайших соседей: {accuracy_score(y_test, y_pred2)}")
print(confusion_matrix(y_test, y_pred2))

# -----------------------------------------------------------------------------
# метод опорных векторов
SVM = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('svm', svm.SVC(C=75)),
     ])
SVM.fit(X_train, y_train)
y_pred3 = SVM.predict(X_test)
with open('model_classification_SVM', 'wb') as picklefile:
    pickle.dump(SVM, picklefile)
print(classification_report(y_test, y_pred3))
print(f"F1 Score  метод опорных векторов: {accuracy_score(y_test, y_pred3)}")
print(confusion_matrix(y_test, y_pred3))