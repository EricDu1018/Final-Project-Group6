import numpy as np
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Normalizing the images :--->
X_train = X_train/255
X_test = X_test/255
print("Done Normalizing!!!")

# Converting the class labels into binary class matrices

y_train = keras.utils.to_categorical(y_train,num_classes=6)
y_test = keras.utils.to_categorical(y_test,num_classes=6)

# Splitting 15% of training dataset into CV dataset
X_train, X_CV, y_train, y_CV = train_test_split(X_train, y_train, test_size=0.15, random_state=0)

# params
drop = 0.5
lrate = 1e-3
batch = 512
epoc = 40

# Creating CNN Network to train the given data
model = Sequential([
    Conv2D(16, (3,3),  activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    AveragePooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    AveragePooling2D((2,2)),
    Flatten(),
    Dense(400, activation="tanh"),
    Dropout(drop),
    BatchNormalization(),
    Dense(6, activation="softmax")
])
# Defining Callback
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=15)


model.compile(optimizer=Adam(lr=lrate), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=batch, epochs=epoc, validation_data=(X_CV,y_CV),callbacks=[early_stopping_callback])

# PERFORMANCE METRIC :-

# ACCURACY ON TEST SET :-
print("Final accuracy on test set : ", 100*model.evaluate(X_test, y_test)[1], "%")

# COHEN KAPPA SCORE ON TEST SET:-
prediction = np.argmax(model.predict(X_test),axis=1)

print("Cohen Kappa score is : ", cohen_kappa_score(prediction,np.argmax(y_test,axis=1)))

# F1 SCORE ON TEST SET :-
print("F1 score is : ", f1_score(prediction,np.argmax(y_test,axis=1), average = 'macro'))

# CONFUSION MATRIX
conf_matrix = confusion_matrix(np.argmax(y_test,axis=1),prediction)

print("Confusion Matrix is : ")

df = pd.DataFrame(conf_matrix,columns=['Buildings','Forest','Glacier','Mountain','Sea','Street'], index=['Buildings','Forest','Glacier','Mountain','Sea','Street'])
sn.set(font_scale=1.4)
sn.heatmap(df,annot=True,fmt='d')
plt.show()