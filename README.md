# BFNNLibrary
A simple python Library for training nerual networks. Mostly for testing out cuda and learning how NNs work.

For usage exmaple check the bottom of BFNNBase.py

the Network Constructor only needs the first and last layer of the Network. All other layers should be referenced directly on indirectly by those layers through self.prevLayer and self.nextLayer
