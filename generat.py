import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import random
import sys

# create the model
model = tf.keras.Sequential()
model.add(layers.Convolution2D(10, kernel_size=(3, 3), strides=(1,1), activation="sigmoid"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation="softmax"))
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

image = cv2.imread("download.jpg")
image = cv2.resize(image, (500, 375))
model.train_on_batch(np.array([image]), np.array([[0,1]]))
model.save_weights("test1.h5")

model.load_weights("epoch 1.h5")


print(model.predict(np.array([image])))