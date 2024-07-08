#only downloads images, cannot train TensorFlow classifier as the page does not exist unfortunately
import tensorflow as tf
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

#we train only the last layer of the neural network to fine tune the existing model for our data