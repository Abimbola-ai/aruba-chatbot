# import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
#nltk.download('wordnet')
#nltk.download("punkt") # uncomment if package is not already downloaded
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
#print(classes)

#lemmatize  and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
#print(words)