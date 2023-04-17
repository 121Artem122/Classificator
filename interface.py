import PySimpleGUI as sg
import stop_words
import string
import re
import pymorphy2
import pickle
from nltk.stem.snowball import SnowballStemmer

russian_stopwords = stop_words.get_stop_words('ru')
russian_stopwords.extend(['здравствуйте', 'до свидания', 'добрый день', 'добрый вечер', 'доброе утро'])


def remove_puctuation(text):
    return ''.join(c for c in text if c not in string.punctuation)


def remove_numbers(text):
    return ' '.join([i if not i.isdigit() else ' ' for i in text])


def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)


def df_preprocess(text):
    reg = re.compile('[^а-яА-яa-zA-Z0-9 ]')  #
    text = text.lower().replace("ё", "е")
    text = text.replace("ъ", "ь")
    text = text.replace("й", "и")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'сайт', text)
    text = re.sub('@[^\s]+', 'пользователь', text)
    text = reg.sub(' ', text)

    stemmer = SnowballStemmer("russian")
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in russian_stopwords])

    return text


def Enter(ent, b):
    now = ent
    adjustment_text = [remove_multiple_spaces(remove_numbers(remove_puctuation(now.lower())))]
    raw_data = adjustment_text
    morph = pymorphy2.MorphAnalyzer()
    tex = []
    for text in raw_data:
        text_lem = [morph.parse(word)[0].normal_form for word in text.split(' ')]
        for word in text_lem:
            if word not in russian_stopwords:
                tex.append(word)
    texor = ' '.join(tex)

    if b == 1:
        with open('model_classification_RandomForest', 'rb') as training_model:
            model = pickle.load(training_model)
            test = model.predict([texor])
    if b == 2:
        with open('model_classification_LogReg', 'rb') as training_model:
            model = pickle.load(training_model)
            test = model.predict([texor])
    return test


sg.theme('GreenTan')
layout = [
    [
        sg.Text("Выберите файл для классификации:", font=("Bodoni MT", 20)),
    ],
    [
        sg.InputText(key='-FILE-'),
        sg.FileBrowse()
    ],
    [sg.Text("Выбор метода классификации", font=("Bodoni MT", 20))],
    [sg.Checkbox('Случайный лес', key='rf', font=("Bodoni MT", 15)),
     sg.Checkbox('Логистическая регрессия', key='lr', font=("Bodoni MT", 15))
     ],
    [sg.Button('Построить модель', font=("Bodoni MT", 15))],
    [sg.Text('Текст принадлежит категории:  ', font=("Bodoni MT", 20)),
     sg.Text('', key='out')
     ]
]

window = sg.Window('Классификатор', layout)

while True:
    event, values = window.read()
    file = values['-FILE-']
    with open(file, 'r') as file1:
        enter = file1.read()
    print(enter)
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    if event == 'Построить модель':
        key_enter = enter
        if values['rf'] == True:
            but = 1
        if values['lr'] == True:
            but = 2
        Answer = str(Enter(enter, but))
        res = Answer.translate({ord(i): None for i in '[\']'})
        window['out'].update(res)
        with open(file, 'w') as file2:
            file2.write(key_enter + '\n' + str(res))
file.close()
window.close()
