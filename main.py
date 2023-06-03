import tkinter as tk
from tkinter import filedialog
import pandas as pd
import re

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from setfit import SetFitClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import neattext.functions as nfx
from nltk.stem import WordNetLemmatizer


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.create_widgets()

    def create_widgets(self):
        # заголовок формы
        self.label = tk.Label(self.master, text="Выберите файл с данными")
        self.label.pack()

        # кнопка "Выбрать файл"
        self.choose_file_button = tk.Button(self.master, text="Выбрать файл", command=self.choose_file)
        self.choose_file_button.pack()

        # кнопка "Обучить классификатор"
        self.train_button = tk.Button(self.master, text="Обучить классификатор", command=self.train_classifier)

        # текстовый блок для вывода результатов
        self.result_text = tk.Text(self.master)
        self.result_text.pack()

        # выпадающий список для выбора модели
        self.model_label = tk.Label(self.master, text="Выберите модель")
        self.model_label.pack()
        self.model_variable = tk.StringVar(self.master)
        self.model_variable.set("paraphrase-MiniLM-L3-v2")
        self.model_dropdown = tk.OptionMenu(self.master, self.model_variable, "paraphrase-MiniLM-L3-v2",
                                            "paraphrase-xlm-r-multilingual-v1", "paraphrase-mpnet-base-v2",
                                            "paraphrase-distilroberta-base-v1")
        self.model_dropdown.pack()

        # текстовое поле для ввода нового текста
        self.new_text_label = tk.Label(self.master, text="Введите новый текст:")
        self.new_text_label.pack()
        self.new_text_entry = tk.Entry(self.master, width=50)
        self.new_text_entry.pack()

        # обработчик события вставки текста в поле ввода
        def handle_paste(event):
            new_text = self.master.clipboard_get()
            self.new_text_entry.insert(tk.INSERT, new_text)

        self.new_text_entry.bind("<Control-v>", handle_paste)

        self.classify_button = tk.Button(self.master, text="Предсказать метку", command=self.classify_text,
                                         state="disabled")
        self.classify_button.pack()

        # создание кнопки для сохранения модели
        self.save_model_button = tk.Button(self.master, text="Сохранить модель", command=self.save_model, state="disabled")
        self.save_model_button.pack(side='left', padx=(10, 0), pady=10)

        # создание кнопки для выбора сохраненной модели
        self.choose_model_button = tk.Button(self.master, text="Выбрать модель", command=self.choose_model)
        self.choose_model_button.pack(side='left', padx=(10, 0), pady=10)

    def choose_file(self):
        # диалог открытия файла
        self.filename = tk.filedialog.askopenfilename(initialdir="/", title="Выберите файл",
                                                      filetypes=(("CSV files", "*.csv"),))

        # вывод пути к файлу на форму
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, f"Выбранный файл: {self.filename}\n")
        self.choose_file_button.config(state="disabled")  # блокируем кнопку "Выбрать файл"
        self.train_button.pack()  # добавляем на форму кнопку "Обучить классификатор"

    def process_text(self, text):
        # приведение к нижнему регистру и удаление знаков препинания
        text = re.sub(r'[^\w\s]', '', text.lower())

        # удаление стоп-слов
        tokens = nfx.remove_stopwords(text, 'en')

        # лемматизация
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens.split()]

        # объединение токенов в строку
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text

    def train_classifier(self):
        # чтение данных из файла
        try:
            data = pd.read_csv(self.filename)
            data = data.sample(frac=1)
            data = data.iloc[:297]
        except Exception as e:
            self.result_text.insert(tk.END, f"Ошибка чтения файла: {e}\n")
            return

        # разделение на тестовую и обучающую выборки
        train_df, test_df = train_test_split(data, test_size=0.3, stratify=data['label'])

        # обработка текста
        train_df['processed_name'] = train_df['name'].apply(self.process_text)

        # ограничение размера обучающей выборки для каждого класса
        categories = data['label'].unique()
        min_size = min([train_df[train_df['label'] == c].shape[0] for c in categories])
        train_df = pd.concat([train_df[train_df['label'] == c].sample(min_size) for c in categories])

        # перемешивание данных
        #train_df = train_df.sample(frac=1)

        # получение выбранной модели
        model = self.model_variable.get()

        # создание и обучение классификатора
        clf = SetFitClassifier(model)
        docs = train_df['processed_name'].values.tolist()
        labels = train_df['label'].values.tolist()
        clf.fit(docs, labels)
        self.clf = clf  # сохраняем модель

        # предсказание меток на тестовых данных
        test_df['predicted'] = clf.predict(test_df['name'].apply(self.process_text).values.tolist())

        # оценка классификатора
        cm = confusion_matrix(test_df['label'], test_df['predicted'])
        report = classification_report(test_df['label'], test_df['predicted'])
        accuracy = accuracy_score(test_df['label'], test_df['predicted'])

        # вывод результатов на форму
        self.result_text.insert(tk.END, f"Точность классификации: {accuracy:.4f}\n\n")
        self.result_text.insert(tk.END, f"Матрица неточностей:\n{cm}\n\n")
        self.result_text.insert(tk.END, "Отчет о классификации:\n")
        self.result_text.insert(tk.END, f"{report}\n\n")
        self.result_text.insert(tk.END, "Первые 5 строк тестовой выборки:\n")
        self.result_text.insert(tk.END, f"{test_df.head()}\n\n")

        self.classify_button.config(state="normal")  # разблокируем кнопку предсказания метки
        self.save_model_button.config(state="normal")# разблокируем кнопку сохранения модели

    def classify_text(self):
        # предсказание метки для нового текста
        new_text = self.new_text_entry.get()

        # обработка текста
        processed_text = self.process_text(new_text)

        # предсказание метки
        label = self.clf.predict([processed_text])[0]

        # вывод результатов на форму
        self.result_text.insert(tk.END, f"Предсказанная метка для текста '{new_text}': {label}\n")
        self.new_text_entry.delete(0, tk.END)
        self.new_text_entry.focus_set()

    def save_model(self):
        # получаем путь к файлу для сохранения модели
        file_path = filedialog.asksaveasfilename(defaultextension='.pkl')

        # сохраняем модель в файл
        with open(file_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def choose_model(self):
        # получаем путь к файлу для загрузки модели
        file_path = filedialog.askopenfilename()

        # загружаем модель из файла
        with open(file_path, 'rb') as f:
            self.clf = pickle.load(f)

        self.result_text.insert(tk.END, f"Выбранный файл: {file_path}\n")
        self.classify_button.config(state="normal")  # разблокируем кнопку предсказания метки

root = tk.Tk()
app = Application(master=root)
app.pack()
app.mainloop()