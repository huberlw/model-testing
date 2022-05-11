from keras.layers import Input, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import math
import cv2
import sys
import os

DATA_DIRECTORY = "./data"
CLASS = ["bobcat", "coyote", "deer", "elk", "human", "not", "raccoon", "weasel"]

SIZE_X = 300
SIZE_Y = 225
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

data, targ = getData(DATA_DIRECTORY, CLASS)
train, train_targ, test, test_targ = split(data, targ, 0.66)

base_model = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(SIZE_X, SIZE_Y, 3)
)

base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalMaxPooling2D(),
        Dense(8, activation='softmax')
    ]
)

"""model = Sequential(
    [
        Input(shape=(SIZE_X, SIZE_Y, COLORS)),
        Conv2D(32, 3, 1, padding='same', activation='relu'),
        Conv2D(64, 3, 1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, 3, 1, padding='same', activation='relu'),
        #Conv2D(256, 3, 1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(8, activation='softmax')
    ]
)"""

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(train, train_targ, epochs=20)
loss, acc = model.evaluate(test, test_targ)

print("RESULTS:")
title = (f"Accuracy: {round(acc, 4)}, Loss: {round(loss, 4)}\n")
print(title)

f = open("ModelTestingData.txt", 'a')
f.write(f"Model: VGG19\nEpochs: 20\nAcc: {acc}\n\n")
f.close()
    
"""plt.plot(history.history['accuracy'], label='accuracy')
plt.title(title)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()"""
