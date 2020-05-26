# Assignment-14_Eva-2

This assignment is about using the TF-records format to reduce the training and execting the model in eager mode

- Famous David net model is used for taraining cifar-10 dataset

What are TF-records :

![tfrecord](https://i.ytimg.com/vi/M7FQIrw2rss/hqdefault.jpg)

To read data efficiently it can be helpful to serialize your data and store it in a set of files (100-200MB each) that can each be read linearly. This is especially true if the data is being streamed over a network. This can also be useful for caching any data-preprocessing.

The TFRecord format is a simple format for storing a sequence of binary records.

Protocol buffers are a cross-platform, cross-language library for efficient serialization of structured data.

Protocol messages are defined by .proto files, these are often the easiest way to understand a message type.

The tf.Example message (or protobuf) is a flexible message type that represents a {"string": value} mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as TFX.

This notebook will demonstrate how to create, parse, and use the tf.Example message, and then serialize, write, and read tf.Example messages to and from .tfrecord files.

> Simply saying TF-records are a sequence of binary records reading which while training is easy compared to normal file format.

__Observations__ :

- On cifar-10 dataset. Using TF-records each epoc had started taking 22 seconds while without TF-records each record take 64 seconds.
- This is really a good optimization over conventional data reading.
- As we have used eager mode entire training is carried out in a for loop.
