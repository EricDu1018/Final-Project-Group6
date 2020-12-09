import os
import cv2
import numpy as np
from random import randint
import keras.utils
from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix


def get_images(path):
    image = []
    for img in os.listdir(path):
        image.append(cv2.resize(cv2.imread(path + img), (150, 150)))
    image = np.array(image)
    return image


def get_classlabel(class_code):
    labels = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 5: 'street', 4: 'sea'}

    return labels[class_code]


# Preparing Train set
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
    label = to_categorical(label, num_classes=6)

    return data, label

x_train, y_train = load_data(os.getcwd() + "/archive/seg_train/")
x_test, y_test = load_data(os.getcwd() + "/archive/seg_test/")
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=0)

# %--------------------------------------------- Training ----------------------------------------------------------
# Resnet
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3

res = applications.ResNet152V2(input_shape=(150, 150, 3), weights='imagenet', include_top=False)
res.trainable = False
print('Resnet pre trained model is loaded ....')

model = Sequential(
    [res,
     Flatten(),
     Dense(400, activation='tanh'),
     Dropout(0.5),
     BatchNormalization(),
     Dense(6, activation='softmax')
     ]
)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=LR),
              metrics=['accuracy'])

_history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_valid, y_valid),callbacks=[early_stopping_callback])

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
plt.savefig(os.getcwd() + '/ResNet_training_result.png')
plt.show()

# %--------------------------------------------Prediction------------------------------------------------------------
pred_images = get_images(os.getcwd() + "/archive/seg_pred/")
fig = plt.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    rnd_number = randint(0, len(pred_images))
    pred_image = np.array([pred_images[rnd_number]])
    pred_class = get_classlabel(model.predict_classes(pred_image)[0])
    #pred_class = np.argmax(model.predict(pred_image), axis=-1)
    pred_prob = model.predict(pred_image).reshape(6)
    for j in range(2):
        if (j%2) == 0:
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plt.Subplot(fig, inner[j])
            ax.bar([0,1,2,3,4,5],pred_prob)
            fig.add_subplot(ax)

fig.show()
plt.savefig(os.getcwd() + '/ResNet_prediction_result.png')

#
# print("Final accuracy on test set:", 100*model.evaluate(x_test, y_test)[1], "%")
#
# # COHEN KAPPA SCORE ON TEST SET:-
# prediction = np.argmax(model.predict(x_test),axis=1)
#
# print("Cohen Kappa score is : ", cohen_kappa_score(prediction,np.argmax(y_test,axis=1)))
#
# # F1 SCORE ON TEST SET :-
# print("F1 score is : ", f1_score(prediction,np.argmax(y_test,axis=1), average = 'macro'))
#
# # CONFUSION MATRIX
# conf_matrix = confusion_matrix(np.argmax(y_test,axis=1),prediction)
#
# # print("Confusion Matrix is : ")
#
# # df = pd.DataFrame(conf_matrix,columns=['Buildings','Forest','Glacier','Mountain','Sea','Street'], index=['Buildings','Forest','Glacier','Mountain','Sea','Street'])
# # sn.set(font_scale=1.4)
# # sn.heatmap(df,annot=True,fmt='d')
# # plt.show()







