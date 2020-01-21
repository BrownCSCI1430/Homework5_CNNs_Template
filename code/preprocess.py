"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
import tensorflow as tf
import hyperparameters as hp
from matplotlib import pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASSES = [""] * hp.category_num
IDX_TO_CLASS = {}

def get_data(data_path, is_vgg, shuffle, augment):
    """ Returns an image data generator which can be iterated
    through for images and corresponding class labels. """

    def preprocess_fn(img):
        if is_vgg:
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.

            if augment and random.random() < 0.3:
                # TODO: Write your own custon data augmentation
                #       procedure that cannot be achieved using
                #       the arguments of ImageDataGenerator().
                #       For example, you can add a small amount of
                #       noise to the image. Make sure to clip
                #       values between 0 and 1 afterwards.
                pass

        return img

    if augment:
        # TODO: Use the arguments of ImageDataGenerator()
        #       to augment the data.
        ds = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_fn)
    else:
        # Don't perform data augmentation here!
        ds = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_fn)

    img_size = 224 if is_vgg else hp.img_size

    ds = ds.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        class_mode='sparse',
        batch_size=hp.batch_size,
        shuffle=shuffle)

    unordered_classes = []
    for dir_name in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir_name)):
            unordered_classes.append(dir_name)

    for img_class in unordered_classes:
        IDX_TO_CLASS[ds.class_indices[img_class]] = img_class
        CLASSES[int(ds.class_indices[img_class])] = img_class

    return ds
