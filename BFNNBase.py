import math
import numpy as np
from numba import cuda
from BFNNDenseLayer import DenseLayer
from BFNNSoftmaxLayer import SoftmaxLayer
from BFNNDropoutLayer import DropoutLayer
from time import time
from BFNNLayers import layer_categories
import pickle as pk
from BFNNActivations import activation_list

def load_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class Network:

    def __init__(self, root_layer=None, result_layer=None):
        self.rootLayer = root_layer
        self.resultLayer = result_layer

        self.output = None
        self.input = None
        self.input_grad = None
        self.training = False
        self.cost = 0

    @classmethod
    def load_from_file(cls, path, rngstates):
        net = cls()
        file = open(path, "rb")

        data = pk.load(file)

        file.close()

        #print(data)

        layer_cnt = data[0][0]

        layer = None

        for i in range(layer_cnt):
            layer_type = data[i+1][0]
            layer_data = data[i+1][1]

            if layer_type == layer_categories["Dense"]:
                layer = DenseLayer.load_from_data(layer_data, net, layer, rngstates)
            elif layer_type == layer_categories["Softmax"]:
                layer = SoftmaxLayer.load_from_data(layer_data, net, layer, rngstates)
            elif layer_type == layer_categories["Dropout"]:
                layer = DropoutLayer.load_from_data(layer_data, net, layer, rngstates)

            if i == 0:
                net.rootLayer = layer
            if i == layer_cnt-1:
                net.resultLayer = layer

        return net

    def forward(self):

        layer = self.rootLayer
        while layer is not None:
            layer.forward(self.input.shape[0])
            layer = layer.nextLayer

        self.output = self.resultLayer.output_d

    def backward(self):

        layer = self.resultLayer
        while layer is not None:
            layer.backward(self.input.shape[0])
            layer = layer.prevLayer

    def compute_cost(self, ground_truth, cost_function):

        self.resultLayer.output_grad_d = cuda.device_array_like(self.output)
        cost_per_batch = cuda.device_array(ground_truth.shape[0], dtype=np.float32)

        if cost_function == "mse":
            compute_cost_gpu_mse[32, 512](self.output, ground_truth, self.resultLayer.output_grad_d, cost_per_batch)
        elif cost_function == "cee":
            compute_cost_gpu_cee[32, 512](self.output, ground_truth, self.resultLayer.output_grad_d, cost_per_batch)

        cuda.synchronize()

        self.cost = np.sum(cost_per_batch.copy_to_host())/ground_truth.shape[0]

    def optimize(self, learning_rate, momentum):

        layer = self.rootLayer
        while layer is not None:
            layer.optimize(learning_rate, momentum)
            layer = layer.nextLayer

    def fit(self, train_input, train_labels, test_input, test_labels, batch_size, epochs, test_accuracy_func, learning_rate=0.1, momentum=0.8, test_equaltity_gpu=False, cost_type="mse", save_iter = 10, start_epoch=0):
        test_labels_d = cuda.to_device(test_labels)
        train_labels_d = cuda.to_device(train_labels)
        train_input_d = cuda.to_device(train_input)
        test_input_d = cuda.to_device(test_input)

        cost_list = []
        acc_list = []

        for epoch in range(start_epoch, start_epoch+epochs):
            epoch_start_time = time()
            print("Epoch", epoch)
            epoch_cost_list = []
            for i in range(0, train_input.shape[0], batch_size):
                #print("Input", i)
                self.training = True

                last_input = min(i+batch_size, train_labels_d.shape[0]-1)

                self.input = train_input_d[i:last_input]

                self.forward()

                self.compute_cost(train_labels_d[i:last_input], cost_type)

                epoch_cost_list.append(self.cost)

                self.backward()
                self.optimize(learning_rate, momentum)

            cost_list.append(np.mean(epoch_cost_list))
            print("Avg. Train Cost:", cost_list[-1])

            self.training = False

            epoch_cost_list = []
            epoch_acc_list = []
            for i in range(0, test_input.shape[0], batch_size):

                last_input = min(i+batch_size, train_labels_d.shape[0]-1)

                self.input = test_input[i:last_input]

                self.forward()

                self.compute_cost(test_labels_d[i:last_input], cost_type)
                if not test_equaltity_gpu:
                    predictions = self.output.copy_to_host()
                else:
                    predictions = self.output
                accuracy = test_accuracy_func(predictions, test_labels)
                epoch_acc_list.append(accuracy)

                epoch_cost_list.append(self.cost)

            #self.input = test_input_d

            #self.forward()
            """
            predictions = None
            if not test_equaltity_gpu:
                predictions = self.output.copy_to_host()
            else:
                predictions = self.output
            accuracy = test_accuracy_func(predictions, test_labels)
            """

            #self.compute_cost(test_labels_d, cost_type)
            print("Avg. Validation Cost:", np.mean(epoch_cost_list))

            acc_list.append(np.mean(epoch_acc_list))
            print("Test Accuracy:", acc_list[-1])
            print("Epoch took", time()-epoch_start_time, "s")

            if (epoch+1)%save_iter == 0:
                self.save("data/network_epoch_" + str(epoch) + ".bin")
                print("Saved to file")

        return cost_list, acc_list

    def layer_count(self):
        cnt = 0
        layer = self.rootLayer
        while layer is not None:
            cnt += 1
            layer = layer.nextLayer
        return cnt

    def save(self, path):
        data_to_save = [[self.layer_count()]]
        layer = self.rootLayer
        while layer is not None:
            layer_type, next_data = layer.get_save_data()
            data_to_save.append([layer_type, next_data])
            layer = layer.nextLayer

        file = open(path, "wb")

        #print(data_to_save)

        pk.dump(data_to_save, file)

        file.close()


@cuda.jit
def compute_cost_gpu_mse(output, ground_truth, output_grad, cost_per_batch):
    for batch_nr in range(cuda.grid(1), ground_truth.shape[0], cuda.gridsize(1)):
        cost_per_batch[batch_nr] = 0
        for node in range(ground_truth.shape[1]):
            cost_per_batch[batch_nr] += (ground_truth[batch_nr, node] - output[batch_nr, node])**2
            output_grad[batch_nr, node] = -(ground_truth[batch_nr, node] - output[batch_nr, node]) * 2

@cuda.jit
def compute_cost_gpu_cee(output, ground_truth, output_grad, cost_per_batch):
    for batch_nr in range(cuda.grid(1), ground_truth.shape[0], cuda.gridsize(1)):
        cost_per_batch[batch_nr] = 0
        for node in range(ground_truth.shape[1]):
            gt = ground_truth[batch_nr, node]
            pred = output[batch_nr, node]
            if gt == 1:
                cost_per_batch[batch_nr] -= math.log(pred)
            elif gt == 0:
                cost_per_batch[batch_nr] -= math.log(1-pred)
            else:
                cost_per_batch[batch_nr] -= gt*math.log(pred) + (1-gt)*math.log(1-pred)
            output_grad[batch_nr, node] = -((gt/pred) - ((1 - gt)/(1 - pred)))


def softmax_prediction_accuracy(predictions, labels):
    acc_sum = 0
    for test_nr in range(test_input.shape[0]):
        max = 0
        idx = 0
        for i in range(10):
            if predictions[test_nr, i] > max:
                max = predictions[test_nr, i]
                idx = i
        if idx == labels[test_nr]:
            acc_sum += 1
    accuracy = acc_sum / test_input.shape[0]
    return accuracy


if __name__ == "__main__":
    rng_states = cuda.random.create_xoroshiro128p_states(256 * 32, seed=2)
    np.random.seed(1)
    batch_size = 128
    epochs = 140

    """
    net = Network()
    l1 = DenseLayer(784, 128, rng_states, activation_list["relu"], net, random_range=0.1)
    l2 = DropoutLayer(128, net, rng_states, 0.3, prev_layer=l1)
    l3 = DenseLayer(128, 10, rng_states, activation_list["linear"], net, random_range=0.1, prev_layer=l2)
    lsm = SoftmaxLayer(10, net, prev_layer=l3)

    net.rootLayer = l1
    net.resultLayer = lsm
    """

    train_input_raw, train_labels_raw = load_mnist("data/fashion", kind="train")

    train_input = train_input_raw/255
    train_labels = np.zeros((train_input.shape[0], 10), dtype=np.float32)

    for i in range(train_input.shape[0]):
        train_labels[i, train_labels_raw[i]] = 1

    test_input_raw, test_labels = load_mnist("data/fashion", kind="t10k")

    test_input = test_input_raw / 255

    net = Network.load_from_file("data/network_epoch_149.bin", rng_states)

    cost_list, acc_list = net.fit(train_input, train_labels, test_input, test_labels, batch_size, epochs, softmax_prediction_accuracy, 0.00001, save_iter=10, start_epoch=150, cost_type="cee")

    print(cost_list)
    print(acc_list)

    net.save("data/clothing_net_final.bin")
