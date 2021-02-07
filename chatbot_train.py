# import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
nltk.download("punkt")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

#create list
words = []
classes = []
documents = []
ignore_letters = ["!", "?", ".", "."] #letters to ignore

intent_file = open("intents.json").read()
intents = json.loads(intent_file)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        word  = nltk.word_tokenize(pattern)
        words.extend(word) # stores the tokenized word in the list words
        # add to documents
        documents.append((word, intent["tag"]))
        # add to classes
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
print(documents)