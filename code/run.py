"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import argparse
import tensorflow as tf
from vgg_model import VGGModel
from your_model import YourModel
import hyperparameters as hp
from preprocess import get_data
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '2'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (2).''')
    parser.add_argument(
        '--load',
        default=os.getcwd() + '/vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 2.''')
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='''Path to model checkpoint directory (remember to
        include the slash at the end of the name).
        Checkpoints are automatically saved when you train your
        model. If you want to continue training from where you
        left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.''')

    return parser.parse_args()

def train(model, train_data, test_data, checkpoint_path):
    """ Training routine. """

    callback_list = []

    # Callback for checkpoint saving
    callback_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path))

    # Callbacks for Tensorboard visualizations
    callback_list.append(tf.keras.callbacks.TensorBoard(
        update_freq='batch',
        profile_batch=0))
    callback_list.append(ImageLabelingLogger(ARGS.data, ARGS.task))

    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(ARGS.data, ARGS.task))

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    model.fit(
        x=train_data,
        validation_data=test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list
    )

def main():
    """ Main function. """

    train_data = get_data(
        os.path.join(ARGS.data, "train"),
        ARGS.task == '2', True, True)
    test_data = get_data(
        os.path.join(ARGS.data, "test"),
        ARGS.task == '2', False, False)

    if ARGS.task == '1':
        model = YourModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        model.summary()
        checkpoint_path = "./your_model_checkpoint/"
    else:
        model = VGGModel()
        checkpoint_path = "./vgg_model_checkpoint/"
        model(tf.keras.Input(shape=(224, 224, 3)))
        model.summary()
        if ARGS.checkpoint is None:
            model.load_weights(ARGS.load, by_name=True)

    if ARGS.checkpoint is not None:
        model.load_weights(ARGS.checkpoint)

    train(model, train_data, test_data, checkpoint_path)

# Make arguments global
ARGS = parse_args()

main()
