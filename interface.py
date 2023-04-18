import PySimpleGUI as sg
import stop_words
import string
import re
import pymorphy2
import pickle
from nltk.stem.snowball import SnowballStemmer
import openpyxl

russian_stopwords = stop_words.get_stop_words('ru')


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


def Enter(ent, b):

    row_data = df_preprocess(ent)

    if b == 1:
        with open('model_classification_RandomForest', 'rb') as training_model:
            model = pickle.load(training_model)
            test = model.predict([row_data])
    if b == 2:
        with open('model_classification_LogReg', 'rb') as training_model:
            model = pickle.load(training_model)
            test = model.predict([row_data])
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

    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    if event == 'Построить модель':

        if values['rf'] == True:
            but = 1
        if values['lr'] == True:
            but = 2

        workbook = openpyxl.load_workbook(file)
        worksheet = workbook.active

        i = 1

        while (i<3000):
            cell_value = worksheet['A' + str(i)].value
            Result = str(Enter(str(cell_value),but))
            worksheet['B'+str(i)] = Result
            i+=1
            print(i)

        workbook.save('result.xlsx')

file.close()
window.close()
