import codecs
import csv
import pickle
import pandas as pd
import re
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


economy = cav_to_list(csv.reader(codecs.open('economy.csv', 'rU', 'utf-8', errors='ignore')))
cooking = cav_to_list(csv.reader(codecs.open('cooking.csv', 'rU', 'utf-8', errors='ignore')))
culture = cav_to_list(csv.reader(codecs.open('culture.csv', 'rU', 'utf-8', errors='ignore')))
sport = cav_to_list(csv.reader(codecs.open('sport.csv', 'rU', 'utf-8', errors='ignore')))

df_economy = pd.DataFrame(economy, columns=['recall'])
df_economy['type'] = 0
df_economy.head()

df_cooking = pd.DataFrame(cooking, columns=['recall'])
df_cooking['type'] = 1
df_cooking.head()

df_culture = pd.DataFrame(culture, columns=['recall'])
df_culture['type'] = 2
df_culture.head()

df_sport = pd.DataFrame(sport, columns=['recall'])
df_sport['type'] = 3
df_sport.head()

raw_data = pd.concat((df_economy, df_cooking, df_sport), axis=0).sample(frac=1.0)
raw_data.head()

russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['здравствуйте', 'до свидания', 'добрый день', 'добрый вечер', 'доброе утро'])
raw_data['recall'] = raw_data['recall'].apply(df_preprocess)

X = raw_data['recall']
y = raw_data['type']
X_train, X_test, y_train, y_test = train_test_split(raw_data['recall'], raw_data['type'],
                                                    random_state=42,
                                                    test_size=0.3)

kmeans = KMeans ( n_clusters = 8, init = 'k-means++', n_init = 'warn', max_iter = 300, tol = 0.0001, verbose = 0, random_state = None, copy_x = True)

# Векторное представление
vectorizer = CountVectorizer()
X_train_BOW = vectorizer.fit_transform(X_train)
X_test_BOW = vectorizer.transform(X_test)

BOW = LogisticRegression(random_state=43, max_iter=200).fit(X_train_BOW, y_train)
y_predict_BOW = BOW.predict(X_test_BOW)
print(f"\nF1 Score BOW: {accuracy_score(y_predict_BOW, y_test)}")

# -----------------------------------------------------------------------
# Линейная регресия
vectorizer1 = TfidfVectorizer()
X_train_TFIDF = vectorizer1.fit_transform(X_train)
X_test_TFIDF = vectorizer1.transform(X_test)

LinearRegression = LogisticRegression(random_state=43, max_iter=200).fit(X_train_TFIDF, y_train)
y_predict_TFIDF = LinearRegression.predict(X_test_TFIDF)
print(f"\nF1 Score TF-IDF: {accuracy_score(y_predict_TFIDF, y_test)}")

# -----------------------------------------------------------------------------
# Векторное представление с биограммами
vectorizer2 = CountVectorizer(ngram_range=(1, 2))
X_train_BOW_bi = vectorizer2.fit_transform(X_train)
X_test_BOW_bi = vectorizer2.transform(X_test)

BOW_bi = LogisticRegression(random_state=0, max_iter=200).fit(X_train_BOW_bi, y_train)
y_predict_BOW_bi = BOW_bi.predict(X_test_BOW_bi)
print(f"\nF1 Score BOW с биграммами: {accuracy_score(y_predict_BOW_bi, y_test)}")

# -----------------------------------------------------------------------------
# Попытка улучшения результата векторного представление с биограммами классификации
lsvc = LinearSVC(C=.5)  # C = 0.5
selective_model = SelectFromModel(lsvc, max_features=None)

X_train_BOW_bi_select_features = selective_model.fit_transform(X_train_BOW_bi, y_train)
X_test_BOW_bi_select_features = selective_model.transform(X_test_BOW_bi)
print('\nNew shapes: ', X_train_BOW_bi.shape, X_test_BOW_bi.shape)
print('\nNew shapes: ', X_train_BOW_bi_select_features.shape, X_test_BOW_bi_select_features.shape)

clf3 = LogisticRegression(random_state=0, max_iter=200)
scores = cross_val_score(clf3, X_train_BOW_bi_select_features, y_train, cv=3, scoring='accuracy')
print(clf3, '\n Cross-validate: ', scores)

clf4 = LogisticRegression(random_state=0, max_iter=200).fit(X_train_BOW_bi_select_features, y_train)
y_predict_BOW_bi = clf4.predict(X_test_BOW_bi_select_features)
print(f"\nF1 Score BOW с биграммами NEWRES: {accuracy_score(y_predict_BOW_bi, y_test)}")

# -----------------------------------------------------------------------------
# Случайный лес
RandomForest = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('clf', RandomForestClassifier(random_state=0)),
])
RandomForest.fit(X_train, y_train)
y_pred = RandomForest.predict(X_test)
with open('model_classification_RandomForest', 'wb') as picklefile:
    pickle.dump(RandomForest, picklefile)
print(classification_report(y_test, y_pred))
print(f"F1 Score случайный лес: {accuracy_score(y_test, y_pred)}")

# -----------------------------------------------------------------------------
# Логистическая регрессия
LogReg = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
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

# -----------------------------------------------------------------------------
# Случайный лес обЪединенный с линейной регресией
RandomForest1 = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(random_state=0)),
])
RandomForest1.fit(X_train, y_train)
y_pred2 = RandomForest1.predict(X_test)
with open('model_classification_RandomForest_1', 'wb') as picklefile:
    pickle.dump(RandomForest1, picklefile)
print(classification_report(y_test, y_pred2))
print(f"F1 Score случайный лес обЪединенный с линейной регресией: {accuracy_score(y_test, y_pred2)}")

# -----------------------------------------------------------------------------
# Логистическая регрессия обЪединенная с линейной регресией
LogReg1 = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=5,
                             max_df=0.7,
                             stop_words=russian_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(n_jobs=3, C=1e5, solver='saga',
                               multi_class='multinomial',
                               max_iter=100,
                               random_state=42)),
])
LogReg1.fit(X_train, y_train)
y_pred3 = LogReg1.predict(X_test)
with open('model_classification_LogReg_1', 'wb') as picklefile:
    pickle.dump(LogReg1, picklefile)
print(classification_report(y_test, y_pred3))
print(f"F1 Score логистическая регрессия обЪединенная с линейной регресией:"
      f"{accuracy_score(y_test, y_pred3)}")
