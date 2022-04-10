import numpy as np

class fc_layer():
    
    def __init__(self, in_dim, out_dim, sigma):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        np.random.seed(252)
        self.W = np.random.normal(0, sigma, size = (self.out_dim, self.in_dim))
        self.b = np.zeros((self.out_dim, 1))
            
    def forward(self, a):
        self.a = a
        self.z = np.dot(self.W, a) + self.b
        return self.z
    
    def backward(self, grad):
        a = self.a
        self.grad_W = np.dot(grad, a.T)
        self.grad_b = grad
        return np.dot(self.W.T, grad)
        
    def update(self, learning_rate, penalty):
        self.W = self.W - learning_rate * (self.grad_W + penalty * self.W)
        self.b = self.b - learning_rate * self.grad_b

class activate_layer():
            
    def forward(self, z):
        self.z = z
        self.a = (1 - np.exp(-2 * z)) / (1 + np.exp(-2 * z))
        return self.a
        
    def backward(self, grad):
        a = self.a
        return (1 - a**2) * grad
    
class softmax_layer():
    
    def forward(self, z):
        self.z = z
        exp_z = np.exp(z)
        self.a = exp_z / np.sum(exp_z)
        return self.a
        
    def backward(self, grad):
        a = self.a
        A = -np.dot(a, a.T)
        A = A + np.diag(a.reshape((-1,)))
        return np.dot(A.T, grad)