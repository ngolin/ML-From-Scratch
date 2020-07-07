import numpy as np
from mlfromscratch.utils import make_diagonal, normalize

# Optimizers for models that use gradient based methods for finding the 
# weights that minimizes the loss.
# A great resource for understanding these methods: 
# https://ruder.io/optimizing-gradient-descent/index.html

class StochasticGradientDescent():
    def __init__(self, learning_rate=0.01, momentum=0, nesterov=False):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.nesterov = nesterov
        self.w_updt = 0

    def update(self, w, grad_wrt_w):
        # https://cs231n.github.io/neural-networks-3/#sgd
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt - self.learning_rate * grad_wrt_w
        # Move against the gradient to minimize loss
        if self.nesterov:
            return w  + self.momentum * self.w_updt - self.learning_rate * grad_wrt_w
        return w + self.w_updt


class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = 0 # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w, grad_wrt_w):
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)
        # Adaptive gradient with higher learning rate for sparse data
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)

class Adadelta():
    def __init__(self, rho=0.95, eps=1e-6):
        self.E_w_updt = 0 # Running average of squared parameter updates
        self.E_grad = 0   # Running average of the squared gradient of w
        self.eps = eps
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # Update average of gradients at w
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)
        
        RMS_delta_w = np.sqrt(self.E_w_updt + self.eps)
        RMS_grad = np.sqrt(self.E_grad + self.eps)

        # Adaptive learning rate
        adaptive_lr = RMS_delta_w / RMS_grad

        # Calculate the update
        w_updt = adaptive_lr * grad_wrt_w

        # Update the running average of w updates
        self.E_w_updt = self.rho * self.E_w_updt + (1 - self.rho) * np.power(w_updt, 2)

        return w - w_updt

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = 0 # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate *  grad_wrt_w / np.sqrt(self.Eg + self.eps)

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = 0
        self.v = 0
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):
        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt
