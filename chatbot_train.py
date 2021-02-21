from data_preprocess import *

# Create our train test
train_x = list(train[:,0])
train_y = list(train[:,1])

# Create or model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu")) # First layer with 128 neurons
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu")) #Second layer with 64 neurons
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax")) # Third layer with neurons == no of intents

# Compile model. 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# #fitting and saving the model 
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)

# print("model created")