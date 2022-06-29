from BFNNActivations import activation, activation_deriv
from BFNNLayers import layer_categories
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32
from BFNNLayers import Layer
import struct

class DenseLayer(Layer):

    def __init__(self, input_size, layer_size, rng_states, activation_idx, network, init_weights=None, init_biases=None, prev_layer=None, random_range=0.01, gpu_settings=(32, 256)):
        self.blocks, self.threads = gpu_settings
        self.layerSize = layer_size
        self.inputSize = input_size
        self.prevLayer = prev_layer
        self.network = network
        self.nextLayer = None
        self.activationIdx = activation_idx
        if prev_layer is not None:
            prev_layer.nextLayer = self
        if init_weights is not None:
            self.weights_d = cuda.to_device(np.float32(init_weights))
        else:
            self.weights_d = cuda.device_array((layer_size, input_size), dtype=np.float32)
            init_weights_gpu[self.blocks, self.threads](rng_states, random_range, input_size, layer_size, self.weights_d)
            cuda.synchronize()
        if init_biases is not None:
            self.biases_d = cuda.to_device(np.float32(init_biases))
        else:
            self.biases_d = cuda.to_device(np.zeros(layer_size, dtype=np.float32))

        self.weights_grad_d = cuda.device_array_like(self.weights_d)
        self.biases_grad_d = cuda.device_array_like(self.biases_d)
        self.output_d = None
        self.output_grad_d = None
        self.preActivation_d = None
        self.preActivation_grad_d = None
        self.weights_change_d = cuda.device_array_like(self.weights_d)
        self.biases_change_d = cuda.device_array_like(self.biases_d)

    @classmethod
    def load_from_data(cls, data, net, prev_layer, rngstates):
        input_size = data[0]
        layer_size = data[1]
        weights = data[2]
        biases = data[3]
        activation_idx = data[4]

        layer = cls(input_size, layer_size, rngstates, activation_idx, net, weights, biases, prev_layer)
        return layer

    def forward(self, batch_size):
        if self.output_d is None:
            self.output_d = cuda.device_array((batch_size, self.layerSize), dtype=np.float32)
            self.preActivation_d = cuda.device_array((batch_size, self.layerSize), dtype=np.float32)
            self.sumPreActivation_d = cuda.device_array(batch_size, dtype=np.float32)
        elif batch_size != self.output_d.shape[0]:
            self.output_d = cuda.device_array((batch_size, self.layerSize), dtype=np.float32)
            self.preActivation_d = cuda.device_array((batch_size, self.layerSize), dtype=np.float32)
            self.sumPreActivation_d = cuda.device_array(batch_size, dtype=np.float32)

        if self.prevLayer is None:
            #(input_size, layer_size, batch_size, input, weights, biases, pre_activation, activation_idx):
            forward_gpu[self.blocks, self.threads](self.inputSize, self.layerSize, batch_size, self.network.input,
                                                   self.weights_d, self.biases_d, self.preActivation_d,
                                                   self.activationIdx)
        else:
            forward_gpu[self.blocks, self.threads](self.inputSize, self.layerSize, batch_size, self.prevLayer.output_d,
                                                   self.weights_d, self.biases_d, self.preActivation_d,
                                                   self.activationIdx)
        cuda.synchronize()

        forward_gpu_activate[self.blocks, self.threads](self.layerSize, batch_size, self.preActivation_d, self.output_d, self.activationIdx)

        cuda.synchronize()

    def backward(self, batch_size):
        if self.preActivation_grad_d is None:
            self.preActivation_grad_d = cuda.device_array((batch_size, self.layerSize), dtype=np.float32)
        elif batch_size != self.preActivation_grad_d.shape[0]:
            self.preActivation_grad_d = cuda.device_array((batch_size, self.layerSize), dtype=np.float32)

        backward_gpu_preactivation[self.blocks, self.threads](self.preActivation_d, self.output_grad_d, batch_size,
                                                              self.preActivation_grad_d, self.layerSize, self.activationIdx)

        cuda.synchronize()

        if self.prevLayer is None:
            self.network.input_grad = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
            backward_gpu[self.blocks, self.threads](batch_size, self.preActivation_grad_d, self.layerSize, self.inputSize,
                                                    self.network.input, self.network.input_grad, self.weights_d,
                                                    self.weights_grad_d, self.biases_grad_d)
        else:
            self.prevLayer.output_grad_d = cuda.device_array((batch_size, self.inputSize), dtype=np.float32)
            backward_gpu[self.blocks, self.threads](batch_size, self.preActivation_grad_d, self.layerSize, self.inputSize,
                                                    self.prevLayer.output_d, self.prevLayer.output_grad_d,
                                                    self.weights_d, self.weights_grad_d, self.biases_grad_d)
        cuda.synchronize()

    def optimize(self, learning_rate, momentum):
        optimize_gpu[self.blocks, self.threads](learning_rate, momentum, self.inputSize, self.layerSize, self.biases_d,
                                                self.biases_grad_d, self.weights_d, self.weights_grad_d,
                                                self.weights_change_d, self.biases_change_d)
        cuda.synchronize()

    def get_save_data(self):
        data = [self.inputSize, self.layerSize, self.weights_d.copy_to_host(),
                self.biases_d.copy_to_host(), self.activationIdx]
        return layer_categories["Dense"], data


@cuda.jit
def optimize_gpu(learning_rate, momentum, input_size, layer_size, biases, biases_grad, weights, weights_grad, weights_change, biases_change):
    for i in range(cuda.grid(1), layer_size, cuda.gridsize(1)):
        biases_change[i] = biases_change[i]*momentum - biases_grad[i]*learning_rate
        biases[i] += biases_change[i]

    work_size = int((layer_size * input_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (layer_size * input_size))
    work_size_pers = work_size
    if cuda.grid(1) < dec_ws_limit: work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    in_cur = int(start / input_size)
    in_prev = int(start % input_size)

    for i in range(work_size_pers):
        weights_change[in_cur, in_prev] = weights_change[in_cur, in_prev]*momentum \
                                              - weights_grad[in_cur, in_prev]*learning_rate
        weights[in_cur, in_prev] += weights_change[in_cur, in_prev]

        in_prev += 1
        if in_prev == input_size:
            in_prev = 0
            in_cur += 1

@cuda.jit
def backward_gpu_preactivation(pre_activation, output_grad, batch_size, pre_activation_grad, layer_size, activation_idx):
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
        pre_activation_grad[batch_nr, in_cur] = output_grad[batch_nr, in_cur] * activation_deriv(pre_activation[batch_nr, in_cur], activation_idx)

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1


@cuda.jit
def backward_gpu(batch_size, pre_activation_grad, layer_size, input_size, output_prev, output_prev_grad, weights, weights_grad, biases_grad):
    for i in range(cuda.grid(1), layer_size, cuda.gridsize(1)):
        biases_grad[i] = 0
        for j in range(batch_size):
            biases_grad[i] += pre_activation_grad[j, i]
        biases_grad[i] /= batch_size

    work_size = int((input_size * batch_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (input_size * batch_size))
    work_size_pers = work_size
    if cuda.grid(1) < dec_ws_limit: work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    batch_nr = int(start / input_size)
    in_prev = int(start % input_size)

    for i in range(work_size_pers):
        output_prev_grad[batch_nr, in_prev] = 0
        for in_cur in range(layer_size):
            output_prev_grad[batch_nr, in_prev] += weights[in_cur, in_prev] * pre_activation_grad[batch_nr, in_cur]

        in_prev += 1
        if in_prev == input_size:
            in_prev = 0
            batch_nr += 1

    work_size = int((input_size * layer_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (input_size * layer_size))
    work_size_pers = work_size
    if cuda.grid(1) < dec_ws_limit: work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    in_cur = int(start / input_size)
    in_prev = int(start % input_size)

    for i in range(work_size_pers):
        weights_grad[in_cur, in_prev] = 0
        for batch_nr in range(batch_size):
            weights_grad[in_cur, in_prev] += pre_activation_grad[batch_nr, in_cur] * output_prev[batch_nr, in_prev]

        weights_grad[in_cur, in_prev] /= batch_size

        in_prev += 1
        if in_prev == input_size:
            in_prev = 0
            in_cur += 1


@cuda.jit
def forward_gpu(input_size, layer_size, batch_size, input, weights, biases, pre_activation, activation_idx):
    work_size = int((layer_size * batch_size) / cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size * cuda.gridsize(1) - (layer_size * batch_size))
    work_size_pers = work_size
    if (cuda.grid(1) < dec_ws_limit): work_size_pers -= 1
    if cuda.grid(1) < dec_ws_limit:
        start = work_size_pers * cuda.grid(1)
    else:
        start = dec_ws_limit * (work_size - 1) + (cuda.grid(1) - dec_ws_limit) * work_size

    batch_nr = int(start / layer_size)
    in_cur = int(start % layer_size)

    for i in range(work_size_pers):
        pre_activation[batch_nr, in_cur] = biases[in_cur]
        for in_prev in range(input_size):
            value = np.float32(input[batch_nr, in_prev] * weights[in_cur, in_prev])
            pre_activation[batch_nr, in_cur] += value

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1

@cuda.jit
def forward_gpu_activate(layer_size, batch_size, pre_activation, output, activation_idx):
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
        output[batch_nr, in_cur] = activation(pre_activation[batch_nr, in_cur], activation_idx)

        in_cur += 1
        if in_cur == layer_size:
            in_cur = 0
            batch_nr += 1


@cuda.jit
def init_weights_gpu(rng_states, random_range, input_size, layer_size, weights):
    work_size = int((input_size*layer_size)/cuda.gridsize(1) + 1)
    dec_ws_limit = (work_size*cuda.gridsize(1) - (input_size*layer_size))
    work_size_pers = work_size
    if(cuda.grid(1)<dec_ws_limit): work_size_pers -= 1
    if cuda.grid(1)<dec_ws_limit:
        start = work_size_pers*cuda.grid(1)
    else:
        start = dec_ws_limit*(work_size-1) + (cuda.grid(1)-dec_ws_limit)*work_size

    in_cur = int(start / input_size)
    in_prev = int(start % input_size)

    for i in range(work_size_pers):

        weights[in_cur, in_prev] = (xoroshiro128p_uniform_float32(rng_states, cuda.grid(1))*2-1)*random_range
        #weights[in_cur, in_prev] = random_range

        in_prev += 1
        if in_prev == input_size:
            in_prev = 0
            in_cur +=1
