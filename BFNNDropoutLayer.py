import numpy as np
from numba import cuda
from BFNNLayers import Layer
from BFNNLayers import layer_categories
from numba.cuda.random import xoroshiro128p_uniform_float32


class DropoutLayer(Layer):

    def __init__(self, input_size, network, rngstates, dropout_rate, prev_layer=None, gpu_settings=(32, 256)):
        self.blocks, self.threads = gpu_settings
        self.inputSize = input_size
        self.prevLayer = prev_layer
        self.network = network
        self.nextLayer = None
        self.rngstates = rngstates
        self.dropoutRate = dropout_rate
        if prev_layer is not None:
            prev_layer.nextLayer = self

        self.output_d = None
        self.output_grad_d = None
        self.dropoutMask = cuda.device_array(input_size, dtype=np.uint8)

    @classmethod
    def load_from_data(cls, data, net, prev_layer, rngstates):
        input_size = data[0]
        dropout_rate = data[1]
        layer = cls(input_size, net, rngstates, dropout_rate, prev_layer)
        return layer

    def forward(self, batch_size):
        self.output_d = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)

        if self.network.training:
            forward_gpu_dropout_random[self.blocks, self.threads](self.inputSize, self.dropoutRate, self.rngstates,
                                                                  self.dropoutMask)
        else:
            forward_gpu_dropout_off[self.blocks, self.threads](self.inputSize, self.dropoutMask)

        if self.prevLayer is None:
            forward_gpu[self.blocks, self.threads](self.inputSize, batch_size, self.network.input, self.output_d,
                                                   self.dropoutMask)
        else:
            forward_gpu[self.blocks, self.threads](self.inputSize, batch_size, self.prevLayer.output_d, self.output_d,
                                                   self.dropoutMask)

        cuda.synchronize()

    def backward(self, batch_size):

        if self.prevLayer is None:
            self.network.input_grad = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
            backward_gpu[self.blocks, self.threads](self.output_grad_d, batch_size, self.network.input_grad,
                                                    self.inputSize, self.dropoutMask)
        else:
            self.prevLayer.output_grad_d = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
            backward_gpu[self.blocks, self.threads](self.output_grad_d, batch_size, self.prevLayer.output_grad_d,
                                                    self.inputSize,  self.dropoutMask)
        cuda.synchronize()

    def optimize(self, learning_rate, momentum):
        pass

    def get_save_data(self):
        data = [self.inputSize, self.dropoutRate]
        return layer_categories["Dropout"], data


@cuda.jit
def backward_gpu(output_grad, batch_size, input_grad, layer_size, dropout_mask):
    work_size = int((layer_size * batch_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (layer_size * batch_size))
    work_size_pers = work_size
    if cuda.grid(1) < dec_ws_limit: work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    batch_nr = int(start / layer_size)
    in_cur = int(start % layer_size)

    for i in range(work_size_pers):
        input_grad[batch_nr, in_cur] = output_grad[batch_nr, in_cur]*dropout_mask[in_cur]

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1


@cuda.jit
def forward_gpu(layer_size, batch_size, input, output, dropout_mask):
    work_size = int((layer_size * batch_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (layer_size * batch_size))
    work_size_pers = work_size
    if cuda.grid(1) < dec_ws_limit: work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    batch_nr = int(start / layer_size)
    in_cur = int(start % layer_size)

    for i in range(work_size_pers):
        output[batch_nr, in_cur] = dropout_mask[in_cur]*input[batch_nr, in_cur]

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1


@cuda.jit
def forward_gpu_dropout_random(input_size, dropout_rate, rngstates, dropout_mask):
    for in_cur in range(cuda.grid(1), input_size, cuda.gridsize(1)):
        random = xoroshiro128p_uniform_float32(rngstates, cuda.grid(1))
        if random > dropout_rate:
            dropout_mask[in_cur] = 1
        else:
            dropout_mask[in_cur] = 0


@cuda.jit
def forward_gpu_dropout_off(input_size, dropout_mask):
    for in_cur in range(cuda.grid(1), input_size, cuda.gridsize(1)):
        dropout_mask[in_cur] = 1
