import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import random
import sys

# relevant classes, i.e., folder names
class_names = ["Cat", "Dog"]

# count epochs outside of class
epoch_number = 0

# create a list than contains how many images are in each folder
num_of_images = []
for element in class_names:
    num_of_images.append(len(os.listdir(element)))

# create the model
model = tf.keras.Sequential()
model.add(layers.Convolution2D(10, kernel_size=(3, 3), strides=(1,1), activation="sigmoid"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(len(class_names), activation="softmax"))
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def create_batch_dict(num_of_images):
    # create a dictionary that contains shuffled lists,
    # so that can choose which images to use for next batch
    batch_dict = {}
    class_number = 0
    for number in num_of_images:
        ls = []
        for i in range(number):
            ls.append(i)
            random.shuffle(ls)
        batch_dict[class_number] = ls
        class_number += 1

    return batch_dict


def train(mini_batch_size, class_names, num_of_images, num_of_epochs):
    global epoch_number
    global model
    epoch_number += 1
    if epoch_number==num_of_epochs:
        model.save_weights("my_model.h5")
        sys.quit()
    batch_dict = create_batch_dict(num_of_images)


    while True:
        # read the images and create a numpy array
        training_images = []
        training_labels = []
        for i in range(len(class_names)):
            missed = 0
            for image_number in batch_dict[i][0:mini_batch_size]:
                image = cv2.imread(class_names[i] + "/" + str(image_number) + ".jpg")
                try:
                    image = cv2.resize(image, (500, 375))
                    training_images.append(image)
                except:
                    missed += 1
                    pass

            ls = [0] * (len(class_names))
            ls[i] = 1
            training_labels.extend([ls]*(mini_batch_size-missed))

            batch_dict[i] = batch_dict[i][mini_batch_size:]

            if mini_batch_size>len(batch_dict[i]):
                print("completed {} epochs".format(epoch_number))
                model.save_weights("epoch {}". format(epoch_number) + ".h5")
                train(mini_batch_size, class_names, num_of_images, num_of_epochs)

        training_images = np.array(training_images)
        training_labels = np.array(training_labels)

        model.train_on_batch(training_images, training_labels)
        print("step")

train(20, class_names, num_of_images, 10)