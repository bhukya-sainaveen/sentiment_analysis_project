# data_processing.py

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def load_data(file_path, delimiter = ",", encoding = 'utf-8'):
    # Function to lead the dataset
    return pd.read_csv(file_path, delimiter = delimiter, encoding = encoding)

def clean_data(df):
    # Remove special symbols for each text
    df['text'] = df.text.apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
    df['text'] = df.text.apply(lambda x: x.lower())
    # Remove stop words for each text
    stop_words = set(stopwords.words('english'))
    df['text_split'] = df.text.apply(lambda x: " ".join([w for w in word_tokenize(x) if w not in stop_words]))
    return df

