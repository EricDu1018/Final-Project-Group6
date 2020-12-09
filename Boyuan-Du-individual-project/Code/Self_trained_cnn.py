
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
import tensorflow.keras.optimizers as Optimizer
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix



def get_images(path):
    image = []
    for img in os.listdir(path):
        image.append(cv2.resize(cv2.imread(path + img), (150, 150)))
    image = np.array(image)
    return image

def load_data(path):
    # load the image
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    Images = []
    Labels = []
    for item in classes:
        for img in os.listdir(path + item + "/"):
            Images.append(cv2.resize(cv2.imread(path + item + "/" + img), (150, 150)))
            Labels.append(item)

    # One-hot-encoded
    label_encoder = LabelEncoder()
    Labels = label_encoder.fit_transform(Labels)
    #Labels = to_categorical(Labels, num_classes=6)

    return shuffle(Images, Labels, random_state=43)  # Shuffle the dataset you just prepared.


def get_classlabel(class_code):
    labels = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 5: 'street', 4: 'sea'}

    return labels[class_code]


x_train, y_train = load_data(os.getcwd() + '/archive/seg_train/') #Extract the training images from the folders.

x_train = np.array(x_train) #converting the list of images to numpy array.
# Normalize the image
x_train = x_train / 255
y_train = np.array(y_train)
print("Shape of Images:", x_train.shape)
print("Shape of Labels:", y_train.shape)


model = Sequential()
model.add(Conv2D(200, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(180, (3, 3), activation='relu'))
model.add(MaxPooling2D(5, 5))
model.add(Conv2D(180, (3, 3), activation='relu'))
model.add(Conv2D(140, (3, 3), activation='relu'))
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(MaxPooling2D(5, 5))
model.add(Flatten())
model.add(Dense(180, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


EPOCHS = 40

_history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.15)

plt.style.use("ggplot") # refined matplotlib
plt.figure()
plt.plot(np.arange(0,EPOCHS),_history.history["loss"],label ="train_loss")
plt.plot(np.arange(0,EPOCHS),_history.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,EPOCHS),_history.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,EPOCHS),_history.history["val_accuracy"],label="val_acc")
plt.title("loss and accuracy")
plt.xlabel("epoch")
plt.ylabel("loss/acc")
plt.legend(loc="best")
plt.savefig(os.getcwd() + '/Self_training_result.png')
plt.show()

x_test, y_test = load_data(os.getcwd() + '/archive/seg_test/')
x_test = np.array(x_test)
y_test = np.array(y_test)
model.evaluate(x_test, y_test, verbose=1)


print("Final accuracy on test set:", 100*model.evaluate(x_test, y_test)[1], "%")

# COHEN KAPPA SCORE ON TEST SET:-
prediction = np.argmax(model.predict(x_test), axis=1)

print("Cohen Kappa score is : ", cohen_kappa_score(prediction, y_test))

# F1 SCORE ON TEST SET :-
print("F1 score is : ", f1_score(prediction, y_test, average = 'macro'))
