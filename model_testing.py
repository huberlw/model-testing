from keras.layers import Input, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
import numpy as np
import math
import cv2
import sys
import os

DATA_DIRECTORY = "./data"
CLASS = ["interesting", "not"]

SIZE_X = 200
SIZE_Y = 150
COLORS = 3

def getData(dir, classes, max_imgs=sys.maxsize):
    images = []
    for c in classes:
        path = os.path.join(dir, c)
        label = c
        cnt = 0;
        for img in os.listdir(path):
            if cnt < max_imgs:
                try:
                    if COLORS == 1:
                        cur_img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    else:
                        cur_img = cv2.imread(os.path.join(path, img))
                    cur_img = cv2.resize(cur_img, (SIZE_X, SIZE_Y))
                    if c == "interesting":
                        images.append([cur_img, 0])
                    else:
                        images.append([cur_img, 1])
                    cnt += 1
                except Exception as e:
                    print(e)
            else:
                break
    train = []
    test = []
    for i in range(len(images)):
        train.append(images[i][0])
        test.append(images[i][1])

    return shuffle(train, test, random_state=int(time.time()))

def split(train, test, percent=0.66):
    size = len(train)
    bar = math.floor(size * percent)
    
    x_train = np.array(train[:bar]) / 255
    x_train = x_train.reshape(-1, SIZE_X, SIZE_Y, COLORS)
    x_test = np.array(test[:bar])
    
    y_train = np.array(train[bar:size]) / 255
    y_train = y_train.reshape(-1, SIZE_X, SIZE_Y, COLORS)
    y_test = np.array(test[bar:size])
    
    return x_train, x_test, y_train, y_test

train, test = getData(DATA_DIRECTORY, CLASS, 861)
x_train, x_test, y_train, y_test, = split(train, test, 0.8)

""" self made model
model = Sequential(
    [
        Input(shape=(SIZE_X, SIZE_Y, 1)),
        Conv2D(32, 3, 1, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, 3, 1, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, 3, 1, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ]
)
#"""

#"""
base_model = InceptionV3(
    weights='imagenet',
    #weights=None,
    include_top=False,
    input_shape=(SIZE_X, SIZE_Y, 3)
)

base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalMaxPooling2D(),
        Dense(1, activation='sigmoid')
    ]
)
#"""

"""
base_model = ResNet50(
    weights='imagenet',
    #weights=None,
    include_top=False,
    input_shape=(SIZE_X, SIZE_Y, 3)
)

base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalMaxPooling2D(),
        Dense(1, activation='sigmoid')
    ]
)
#"""

"""
base_model = VGG19(
    weights='imagenet',
    #weights=None,
    include_top=False,
    input_shape=(SIZE_X, SIZE_Y, 3)
)

base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalMaxPooling2D(),
        Dense(1, activation='sigmoid')
    ]
)
#"""
        
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(x_train, x_test, epochs=5)
loss, acc = model.evaluate(y_train, y_test)

print("RESULTS:")
title = (f"Accuracy: {round(acc, 3)}, Loss: {round(loss, 3)}\n")
print(title)

plt.plot(history.history['accuracy'], label='accuracy')
plt.title(title)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
