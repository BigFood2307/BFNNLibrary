from BFNNActivations import softmax, softmax_deriv, softmax_deriv_cross
import numpy as np
from numba import cuda
from BFNNLayers import Layer
from BFNNLayers import layer_categories
import math


class SoftmaxLayer(Layer):

    def __init__(self, input_size, network, prev_layer=None, gpu_settings=(32, 256)):
        self.blocks, self.threads = gpu_settings
        self.inputSize = input_size
        self.prevLayer = prev_layer
        self.network = network
        self.nextLayer = None
        if prev_layer is not None:
            prev_layer.nextLayer = self

        self.output_d = None
        self.output_grad_d = None
        self.sumPreActivation_d = None
        self.preActivation_d = None

    @classmethod
    def load_from_data(cls, data, net, prev_layer, rngstates):
        input_size = data[0]

        layer = cls(input_size, net, prev_layer)
        return layer

    def forward(self, batch_size):
        self.preActivation_d = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
        self.output_d = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
        self.sumPreActivation_d = cuda.device_array(batch_size, dtype=np.float32)

        if self.prevLayer is None:
            forward_gpu_pre_activation[self.blocks, self.threads](self.inputSize, batch_size, self.network.input,
                                                                  self.preActivation_d)
        else:
            forward_gpu_pre_activation[self.blocks, self.threads](self.inputSize, batch_size, self.prevLayer.output_d,
                                                                  self.preActivation_d)

        cuda.synchronize()
        forward_gpu_sum[self.blocks, self.threads](self.inputSize, batch_size, self.preActivation_d,
                                                   self.sumPreActivation_d)
        cuda.synchronize()
        forward_gpu_activate[self.blocks, self.threads](self.inputSize, batch_size, self.preActivation_d,
                                                        self.output_d, self.sumPreActivation_d)
        cuda.synchronize()

    def backward(self, batch_size):

        if self.prevLayer is None:
            self.network.input_grad = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
            backward_gpu_preactivation[self.blocks, self.threads](self.preActivation_d, self.output_grad_d, batch_size,
                                                                  self.network.input_grad, self.inputSize,
                                                                  self.sumPreActivation_d)
        else:
            self.prevLayer.output_grad_d = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
            backward_gpu_preactivation[self.blocks, self.threads](self.preActivation_d, self.output_grad_d, batch_size,
                                                                  self.prevLayer.output_grad_d, self.inputSize,
                                                                  self.sumPreActivation_d)
        cuda.synchronize()

    def optimize(self, learning_rate, momentum):
        pass

    def get_save_data(self):
        data = [self.inputSize]
        return layer_categories["Softmax"], data


@cuda.jit
def backward_gpu_preactivation(pre_activation, output_grad, batch_size, pre_activation_grad, layer_size, sum_pre_activation):
    work_size = int((layer_size * batch_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (layer_size * batch_size))
    work_size_pers = work_size
    if cuda.grid(1) < dec_ws_limit: work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    batch_nr = int(start / layer_size)
    in_prev = int(start % layer_size)

    for i in range(work_size_pers):
        pre_activation_grad[batch_nr, in_prev] = 0
        for in_cur in range(layer_size):
            if in_cur == in_prev:
                pre_activation_grad[batch_nr, in_prev] += output_grad[batch_nr, in_cur] * softmax_deriv(pre_activation[batch_nr, in_cur], sum_pre_activation[batch_nr])
            else:
                pre_activation_grad[batch_nr, in_prev] += output_grad[batch_nr, in_cur] * softmax_deriv_cross(pre_activation[batch_nr, in_cur], pre_activation[batch_nr, in_prev], sum_pre_activation[batch_nr])

        #print(sum_pre_activation[batch_nr], pre_activation_grad[batch_nr, in_prev])

        in_prev += 1
        if in_prev == layer_size:
            in_prev = 0
            batch_nr += 1


@cuda.jit
def forward_gpu_pre_activation(layer_size, batch_size, input, pre_activation):
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
        pre_activation[batch_nr, in_cur] = math.exp(input[batch_nr, in_cur])

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1


@cuda.jit
def forward_gpu_sum(input_size, batch_size, input, sum_pre_activation):
    for batch_nr in range(cuda.grid(1), batch_size, cuda.gridsize(1)):
        sum_pre_activation[batch_nr] = 0
        for node in range(input_size):
            sum_pre_activation[batch_nr] += input[batch_nr, node]


@cuda.jit
def forward_gpu_activate(layer_size, batch_size, pre_activation, output, sum_pre_activation):
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
        output[batch_nr, in_cur] = softmax(pre_activation[batch_nr, in_cur], sum_pre_activation[batch_nr])

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1