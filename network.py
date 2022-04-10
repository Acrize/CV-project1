from layer import fc_layer
import numpy as np
from math import pi
import matplotlib.pyplot as plt

def log_loss(pred, label):
    temp = 1 - pred
    temp[label - 1, 0] = 1 - temp[label - 1, 0]
    value = -np.sum(np.log(temp))
    temp[label - 1, 0] = -temp[label - 1, 0]
    grad = 1 / temp
    return value, grad

class fc_network():
    
    def __init__(self, layers):
        self.layers = layers
        self.error = []
        self.loss = []
        self.iter = []
        
    def train(self, max_iter, learning_rate, data, valid=True, print_error_rate=True, penalty=0):
        
        for iter in range(max_iter):
            # retrieve data
            sample, label = data.get_random_data()
            
            # forward
            pred = sample
            for layer in self.layers:
                pred = layer.forward(pred)
                
            # compute loss
            loss, grad = log_loss(pred, label)
                
            # backward
            for layer in self.layers[::-1]:
                grad = layer.backward(grad)
                
            # update
            alpha = learning_rate * (1 + np.cos(iter * pi / max_iter)) / 2
            for layer in self.layers:
                if isinstance(layer, fc_layer):
                    layer.update(alpha, penalty)
                        
            # validation
            if (iter <= 10000 and iter % 1000 == 0) or (iter % 10000 == 0):
                error, loss = self.predict(data, 'valid')
                self.error.append(error)
                self.loss.append(loss)
                self.iter.append(iter)

                print('iters: %5d'%iter)
                if print_error_rate:
                    print('valid error rate: %.5f'%error)
                    print('average loss: %.5f'%loss)
                
        # test
        if valid:
            error, loss = self.predict(data, 'valid')
            print('valid error rate: %.5f'%error)
            print('average loss: %.5f'%loss)
        else:
            error, loss = self.predict(data, 'test')
            print('test error rate: %.5f'%error)
            
        return error
    
    def test(self, data):
        error, loss = self.predict(data, 'test')
        print('test error rate: %.5f'%error)
                
    def predict(self, data, type):
        if type == 'test':
            samples = data.test_set['sample']
            labels = data.test_set['label']
        elif type == 'valid':
            samples = data.valid_set['sample']
            labels = data.valid_set['label']
        
        error = 0
        loss = 0
        for i in range(samples.shape[0]):
            sample = samples[i : (i + 1), :].T
            label = labels[i, 0]
        
            pred = sample
            for layer in self.layers:
                pred = layer.forward(pred)
            
            loss = loss + log_loss(pred, label)[0]
            
            pred = (np.argmax(pred, axis=0) + 1).reshape((1))[0]
            error = error + int(pred != label)
        
        error = error / samples.shape[0]
        loss = loss / samples.shape[0]
        return error, loss
    
    def visualize_error(self):
        iter = self.iter
        error = self.error
        
        plt.plot(iter, error, label='Validation Error', linewidth=2, color='green', 
                 marker='s', markersize=4)
        plt.xlabel('iteration')
        plt.ylabel('error rate')
        plt.title('Validation Error Rate')
        plt.legend()
        plt.grid()
        plt.show()
        
    def visualize_loss(self):
        iter = self.iter
        loss = self.loss
        
        plt.plot(iter, loss, label='Validation Loss', linewidth=2, color='blue', 
                 marker='^', markersize=4)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()
        
    def visualize_weight(self):
        fc_layer_1 = self.layers[0]
        fc_layer_2 = self.layers[2]
        
        visualize_num = 9
        L = 256
        figure_size = 28
        
        np.random.seed(252)
        
        # 1st fc layer
        index = np.arange(fc_layer_1.out_dim)
        np.random.shuffle(index)
        index = index[:visualize_num]
        
        for i in range(visualize_num):
            weight = fc_layer_1.W[index[i], :].copy()
            weight = ((weight - np.min(weight)) / (np.max(weight) - np.min(weight)) * (L - 1)).astype(np.uint8)
            weight_matrix = weight.reshape((figure_size, figure_size))
            
            plt.subplot(round(np.sqrt(visualize_num)), round(np.sqrt(visualize_num)), i + 1)
            plt.imshow(weight_matrix)
            plt.xticks([])
            plt.yticks([])
        
        plt.show()
        
        # 2nd fc layer
        index = np.arange(fc_layer_2.out_dim)
        np.random.shuffle(index)
        index = index[:visualize_num]
        
        for i in range(visualize_num):
            weight = fc_layer_2.W[index[i], :].copy()
            weight = np.dot(weight, fc_layer_1.W)
            weight = ((weight - np.min(weight)) / (np.max(weight) - np.min(weight)) * (L - 1)).astype(np.uint8)
            weight_matrix = weight.reshape((figure_size, figure_size))
            
            plt.subplot(round(np.sqrt(visualize_num)), round(np.sqrt(visualize_num)), i + 1)
            plt.imshow(weight_matrix)
            plt.xticks([])
            plt.yticks([])
        
        plt.show()