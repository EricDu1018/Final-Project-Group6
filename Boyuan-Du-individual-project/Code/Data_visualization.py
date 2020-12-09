import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
from random import randint
from sklearn.preprocessing import LabelEncoder


train_dir = '/Users/ericdu/Desktop/archive/seg_train/'
test_dir = '/Users/ericdu/Desktop/archive/seg_test/'
pred_dir = '/Users/ericdu/Desktop/archive/seg_pred/'

album_train = pd.DataFrame(columns = ["id", "label"])
album_test = pd.DataFrame(columns = ["id", "label"])
dirs = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
for direc in dirs:
    for file in os.listdir(train_dir + direc):
        album_train = album_train.append({"id":file, "label":direc},ignore_index=True)
    for file in os.listdir(test_dir + direc):
        album_test = album_test.append({"id":file, "label":direc},ignore_index=True)

sns.countplot(x="label", data=album_train)
sns.countplot(x="label", data=album_test)


def load_data(path):
    # load the image
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    data = []
    label = []
    for item in classes:
        for img in os.listdir(path + item + "/"):
            data.append(cv2.resize(cv2.imread(path + item + "/" + img), (150, 150)))
            label.append(item)
    data = np.array(data)
    label = np.array(label)

    # Normalize the image
    data = data / 255

    # One-hot-encoded
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    # label = to_categorical(label, num_classes=6)

    return data, label

Images, Labels = load_data(train_dir)


def get_classlabel(class_code):
    labels = {2: 'glacier', 4: 'sea', 0: 'buildings', 1: 'forest', 5: 'street', 3: 'mountain'}

    return labels[class_code]


f, ax = plt.subplots(5, 5)
f.subplots_adjust(0, 0, 3, 3)
for i in range(0, 5, 1):
    for j in range(0, 5, 1):
        rnd_number = randint(0, len(Images))
        ax[i, j].imshow(Images[rnd_number])
        ax[i, j].set_title(get_classlabel(Labels[rnd_number]))
        ax[i, j].axis('off')