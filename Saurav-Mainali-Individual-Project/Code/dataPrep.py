import os
import cv2
import numpy as np

# {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 }

# Preparing Train set

order = [('buildings',0),('forest',1),('glacier',2),('mountain',3),('sea',4),('street',5)]

x_train = []
y_train = []
x_test = []
y_test = []

def prepare_data(training = True):
    if training == 1:
        global x_train
        global y_train
        for scene,label in order:
            for img in os.listdir("seg_train/seg_train/" + scene +"/"):
                x_train.append(cv2.resize(cv2.imread("seg_train/seg_train/" + scene + "/" + img), (150, 150)))
                y_train.append(label)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    else:
        global x_test
        global y_test
        for scene, label in order:
            for img in os.listdir("seg_test/seg_test/" + scene + "/"):
                x_test.append(cv2.resize(cv2.imread("seg_test/seg_test/" + scene + "/" + img),(150, 150)))
                y_test.append(label)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

# Creating TrainSet
prepare_data()

# Creating TestSet
prepare_data(training=False)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.save("X_train.npy", x_train); np.save("y_train.npy", y_train)

np.save("X_test.npy", x_test); np.save("y_test.npy", y_test)

print("Done! Data is prepared.")







