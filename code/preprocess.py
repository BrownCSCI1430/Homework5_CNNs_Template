"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import tensorflow as tf
import hyperparameters as hp
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
        #       to augment the data. Leave the
        #       preprocessing_function argument as is.
        ds = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_fn)
    else:
        # Don't perform augmentation here!
        ds = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_fn)

    img_size = 224 if is_vgg else hp.img_size

    classes_for_flow = None

    # If dictionary is not empty
    if bool(IDX_TO_CLASS):
        classes_for_flow = CLASSES

    # Form imgae data generator from directory structure
    ds = ds.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        class_mode='sparse',
        batch_size=hp.batch_size,
        shuffle=shuffle,
        classes=classes_for_flow)

    # If the dictionary is empty
    if not bool(IDX_TO_CLASS):
        unordered_classes = []
        for dir_name in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, dir_name)):
                unordered_classes.append(dir_name)

        for img_class in unordered_classes:
            IDX_TO_CLASS[ds.class_indices[img_class]] = img_class
            CLASSES[int(ds.class_indices[img_class])] = img_class

    return ds
