"""
This script trains a TensorFlow neural net model on Fashion MNIST data across multiple worker
nodes. Part of this script is based on https://www.tensorflow.org/tutorials/keras/classification

Written by Courtney Pacheco for Red Hat, Inc. 2020.
"""

from __future__ import print_function
from datetime import datetime
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow soft device placement
tf.config.set_soft_device_placement(True)

# Set min/max values of specific vars
EPOCHS_MIN = 1
EPOCHS_MAX = 100
NEURONS_MIN = 10
NEURONS_MAX = 1000
BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 100
NUM_WORKERS_MIN = 1
NUM_WORKERS_MAX = 10
MAX_ATTEMPTS=10

class FashionMNISTNeuralNet:

    def __init__(self, num_epochs=100, num_neurons=10, batch_size=32, num_workers=None):

        # Set multiworker strategy so that we can run across multiple nodes
        self.multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        # Set the number of epochs
        self.num_epochs = num_epochs

        # Set the number of "neurons" (nodes) for the neural net model
        self.num_neurons = num_neurons

        # Set number of nodes in the 'softmax' layer
        self.softmax = 10

        # Set batch size
        self.batch_size = batch_size

        # Set number of worker nodes
        self.num_workers = num_workers

        # Initialize the dataset
        self.dataset = self.__load_dataset()

        # Preprocess the dataset
        self.__preprocess_dataset()

        # Initialize the model
        self.model = None


    def __load_dataset(self):
        """
        Loads the fashion MNIST dataset

        Returns
        -------
        dataset: dict
            A dictionary which contains the training data, testing data, training labels, and
            test labels 
        """
        # Load the data
        attempt = 0
        fashion_mnist = keras.datasets.fashion_mnist
        while attempt < MAX_ATTEMPTS:
            try:
                train_images = None
                train_labels = None
                test_images = None
                test_labels = None
                (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
                if train_images != None and train_labels != None and test_images != None and test_labels != None:
                    break
            except:
                print('Could not load MNIST data. Trying again until we can load it...')

            attempt += 1

        # Pack into a dictionary
        dataset = {'train': {'data': train_images, 'labels': train_labels}, 'test': {'data': test_images, 'labels': test_labels}}

        return dataset


    def __preprocess_dataset(self):
        """
        Preprocesses the fashion MNIST dataset for Neural Networks

        Returns
        -------
        dataset: dictionary
            The original dataset, but preprocessed
        """

        # We need to scale the image data to a [0,1] scale. We have the data in RGB 0-255 format
        scale_factor = 255.0

        # Adjust the training labels so they're 32-bit integers
        train_labels = self.dataset['train']['labels']
        train_labels = train_labels.astype(np.int32)

        # Do the same with the testing labels
        test_labels = self.dataset['test']['labels']
        test_labels = test_labels.astype(np.int32)

        # Grab the training and testing data, then scale
        train_images = self.dataset['train']['data'] / scale_factor
        test_images = self.dataset['test']['data'] / scale_factor

        # Save the preprocessed data back to the dataset
        self.dataset['train']['data'] = train_images
        self.dataset['test']['data'] = test_images
        self.dataset['train']['labels'] = train_labels
        self.dataset['test']['labels'] = test_labels


    def train(self):
        """
        Trains the neural network model
        """

        # Check number of workers
        if self.num_workers is None:
            raise ValueError('Invalid number of workers. Please choose a value greater than or equal to 1.')

        if self.num_workers < 1:
            raise ValueError('Invalid number of workers. Please choose a value greater than or equal to 1.')

        # Use multiple workers
        with self.multiworker_strategy.scope():

            # Extract data from dict and convert to numpy arrays
            data = np.array(self.dataset['train']['data'])
            labels = np.array(self.dataset['train']['labels'])

            # Get image width and height
            image_height = data.shape[1]
            image_width = data.shape[2]

            # Convert data to dataset
            input_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            if self.num_workers > 1:
                input_dataset = input_dataset.repeat(self.num_workers-1).shuffle(5000).batch(self.batch_size)
            else:
                input_dataset = input_dataset.shuffle(5000).batch(self.batch_size)

            # Define model
            self.model = keras.Sequential([
                            keras.layers.Flatten(input_shape=(image_height, image_width)),
                            keras.layers.Dense(self.num_neurons),
                        ])

            # Compile model
            self.model.compile(optimizer='adam',
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               metrics=['accuracy'])

            print('Training...')
            start_train = datetime.now()
            self.model.fit(input_dataset, epochs=self.num_epochs, steps_per_epoch=self.num_epochs)
            finish_train = datetime.now()

            # Calculate elapsed time
            elapsed_time_seconds = (finish_train - start_train).total_seconds()

            # Print elapsed time
            print('    Train time (sec):', elapsed_time_seconds)


    def test(self):
        """
        Tests the neural network model
        """
        # Check if we have already trained
        if self.model is None:
            raise AttributeError('Model has not been trained yet. Please train the model first.')

        # Use multiple workers
        with self.multiworker_strategy.scope():

            # Extract dataset
            data = np.array(self.dataset['test']['data'])
            labels = np.array(self.dataset['test']['labels'])

            # Convert to Dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            if self.num_workers > 1:
                test_dataset = test_dataset.repeat(self.num_workers-1).shuffle(5000).batch(self.batch_size)
            else:
                test_dataset = test_dataset.shuffle(5000).batch(self.batch_size)

            print('Testing...')
            start_test = datetime.now()
            test_loss, test_acc = self.model.evaluate(test_dataset, steps=self.num_epochs, verbose=2)
            finish_test = datetime.now()

            # Calculate elapsed time
            elapsed_time_seconds = (finish_test - start_test).total_seconds()

            # Print elapsed time
            print('    Test time (sec):', elapsed_time_seconds)

            # Print loss and accuracy
            print('    Loss:', test_loss)
            print('    Accuracy:', test_acc)


def run_mnist(num_epochs, num_neurons, batch_size, num_workers):
    """
    Runs the fashion MNIST training and classification neural network

    Parameters
    ----------
    num_epochs: int
        Number of epochs to use when training (default: 100)

    num_neurons: int
        Number of neurons (nodes) to use in the first layer (default: 128)

    batch_size: int
        Training batch size (default: 32)

    num_workers: int
        Number of OpenShift/Kubernetes workers to be used
    """

    # Define the neural network
    neural_net = FashionMNISTNeuralNet(num_epochs, num_neurons, batch_size, num_workers)

    # Train
    neural_net.train()

    # Test
    neural_net.test()


if __name__ == '__main__':

    # Make sure the user passed in all 3 arguments
    if len(sys.argv) < 5:
        raise RuntimeError('Too few arguments provided. Please provide four arguments: (1.) number of epochs, (2.) number of neurons (nodes), (3.) batch size, and (4.) number of workers.')

    if len(sys.argv) > 5:
        raise RuntimeError('Too many arguments provided. Please provide four arguments: (1.) number of epochs, (2.) number of neurons (nodes), (3.) batch size, and (4.) number of workers.')

    # Convert arguments to integers
    num_epochs = int(sys.argv[1])
    num_neurons = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    num_workers = int(sys.argv[4])

    # Check the values of the arguments, making sure they're within the acceptable range
    error_msg_template = 'Number of %s must be in the range of [%d, %d]. You entered: %d.'
    errors = []
    if num_epochs < EPOCHS_MIN or num_epochs > EPOCHS_MAX:
        error_msg = error_msg_template % ('epochs', EPOCHS_MIN, EPOCHS_MAX, num_epochs)
        errors.append(error_msg)

    if num_neurons < NEURONS_MIN or num_neurons > NEURONS_MAX:
        error_msg = error_msg_template % ('neurons', NEURONS_MIN, NEURONS_MAX, num_neurons)
        errors.append(error_msg)

    if num_workers < NUM_WORKERS_MIN or num_workers > NUM_WORKERS_MAX:
        error_msg = error_msg_template % ('worker nodes', NUM_WORKERS_MIN, NUM_WORKERS_MAX, num_workers)
        errors.append(error_msg)

    if batch_size < BATCH_SIZE_MIN or batch_size > BATCH_SIZE_MAX:
        error_msg = 'Batch size must be in the range of [%d, %d]. You entered: %d' % (BATCH_SIZE_MIN, BATCH_SIZE_MAX, batch_size)
        errors.append(error_msg)

    if len(errors) > 0:
        all_errors = ''
        for msg in errors:
            all_errors += (msg + ' ')
        raise ValueError(all_errors)

    # Run the MNIST classification
    run_mnist(num_epochs, num_neurons, batch_size, num_workers)
