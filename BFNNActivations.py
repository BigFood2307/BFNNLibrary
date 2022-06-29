from numba import cuda
import math

activation_list = dict()
activation_list["sigmoid"] = 0
activation_list["relu"] = 1
activation_list["linear"] = 2
activation_list["dorelu"] = 3


@cuda.jit(device=True)
def activation(x, a):
    if a == 0: return sigmoid(x)
    elif a == 1: return relu(x)
    elif a == 2: return linear(x)
    elif a == 3: return dorelu(x)
    return 0


@cuda.jit(device=True)
def activation_deriv(x, a):
    if a == 0: return sigmoid_deriv(x)
    elif a == 1: return relu_deriv(x)
    elif a == 2: return linear_deriv(x)
    elif a == 3: return dorelu_deriv(x)
    return 0


@cuda.jit(device=True)
def sigmoid(x):
    return 1/(1+math.exp(-x))


@cuda.jit(device=True)
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


@cuda.jit(device=True)
def relu(x):
    return max(0, x)


@cuda.jit(device=True)
def relu_deriv(x):
    return int(x > 0)


@cuda.jit(device=True)
def linear(x):
    return x


@cuda.jit(device=True)
def linear_deriv(x):
    return 1


@cuda.jit(device=True)
def dorelu(x):
    return max(0, min(1, x))


@cuda.jit(device=True)
def dorelu_deriv(x):
    return int((x > 0) and (x < 1))


@cuda.jit(device=True)
def softmax(x, sum_x):
    return x/sum_x


@cuda.jit(device=True)
def softmax_deriv(x, sum_x):
    return x/sum_x - (x**2)/(sum_x**2)


@cuda.jit(device=True)
def softmax_deriv_cross(x_str, x_cr, sum_x):
    return -(x_str*x_cr)/(sum_x**2)



