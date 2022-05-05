from keras.layers import Input, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
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
        label = CLASS.index(c)
        cnt = 0;
        for img in os.listdir(path):
            if cnt < max_imgs:
                try:
                    if COLORS == 1:
                        cur_img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    else:
                        cur_img = cv2.imread(os.path.join(path, img)) 
                    cur_img = cv2.resize(cur_img, (SIZE_X, SIZE_Y))
                    images.append([cur_img, label])
                    cnt += 1
                except Exception as e:
                    print(e)
            else:
                break
    imgs = []
    targ = []
    for i in range(len(images)):
        imgs.append(images[i][0])
        targ.append(images[i][1])

    return shuffle(imgs, targ)

def split(data, targ, percent=0.66):
    size = len(data)
    bar = math.floor(size * percent)
    
    train = np.array(data[:bar]) / 255
    train = train.reshape(-1, SIZE_X, SIZE_Y, COLORS)
    train_targ = np.array(targ[:bar])
    
    test = np.array(data[bar:size]) / 255
    test = test.reshape(-1, SIZE_X, SIZE_Y, COLORS)
    test_targ = np.array(targ[bar:size])
    
    return train, train_targ, test, test_targ

data, targ = getData(DATA_DIRECTORY, CLASS, 861)
train, train_targ, test, test_targ = split(data, targ, 0.66)

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(SIZE_X, SIZE_Y, 3)
)

base_model.trainable = False

model = Sequential(
    [
        base_model,
        #GlobalMaxPooling2D(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)
        
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(train, train_targ, epochs=5)
loss, acc = model.evaluate(test, test_targ)

print("RESULTS:")
title = (f"Accuracy: {round(acc, 4)}, Loss: {round(loss, 4)}\n")
print(title)

plt.plot(history.history['accuracy'], label='accuracy')
plt.title(title)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.show()
