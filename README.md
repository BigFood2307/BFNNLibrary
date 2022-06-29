# BFNNLibrary
A simple python Library for training nerual networks. Mostly for testing out cuda and learning how NNs work.

For usage example check the bottom of BFNNBase.py

The Network Constructor only needs the first and last layer of the Network. All other layers should be referenced directly on indirectly by those layers through self.prevLayer and self.nextLayer

To use the Mnist (Fashion) Dataset, place the t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz in a folder and adjust the path for load_mnist()
Default folder is data/fashion