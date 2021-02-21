# import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk # natural language tool kit
# nltk.download('wordnet')
# nltk.download("punkt") # uncomment if package is not already downloaded
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# create list
words = []
classes = []
documents = []
ignore_letters = ["!", "?", ".", ","] # letters to ignore

intent_file = open("intents.json").read()
intents = json.loads(intent_file)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        word  = nltk.word_tokenize(pattern)
        words.extend(word) # stores the tokenized word in the list words
        # add to documents
        documents.append((word, intent["tag"])) # combination of word and intent-tag
        # add to classes
        if intent["tag"] not in classes:  # appended only intent["tag"]
            classes.append(intent["tag"])
# print(classes)
# print(documents)
# print(words)
#lemmatize  and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters] #https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
words = sorted(list(set(words)))
# print(words)

# sort the classes
classes = sorted(list(set(classes))) # set helps us to create unique elements

# print(len(documents), "documents")
# print(len(classes), "classes", classes)
# print(len(words), "unique lemmatized word", words)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# create our training data
train = []

#create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    bag_of_words = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] =  1
    
    train.append([bag_of_words, output_row])


random.shuffle(train)
train = np.array(train, dtype=object)

# print(train)
