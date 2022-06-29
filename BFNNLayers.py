from numba import cuda
from abc import ABC, abstractmethod

layer_categories = dict()
layer_categories["Dense"] = 0
layer_categories["Softmax"] = 1
layer_categories["Dropout"] = 2


class Layer(ABC):

    def __init__(self, output_shape):
        self.output_d = cuda.device_array(output_shape)
        self.prevLayer = None
        self.nextLayer = None

    @abstractmethod
    def forward(self, batch_size):
        pass

    @abstractmethod
    def backward(self, batch_size):
        pass

    @abstractmethod
    def optimize(self, learning_rate, momentum):
        pass

    @abstractmethod
    def get_save_data(self):
        pass

