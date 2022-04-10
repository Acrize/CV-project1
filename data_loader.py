import gzip
import struct
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

class data_loader():
    
    def __init__(self):
        
        train_path = osp.join('data', 'train')
        train_sample_path = osp.join(train_path, 'train-images-idx3-ubyte.gz')
        train_label_path = osp.join(train_path, 'train-labels-idx1-ubyte.gz')
        test_path = osp.join('data', 'test')
        test_sample_path = osp.join(test_path, 't10k-images-idx3-ubyte.gz')
        test_label_path = osp.join(test_path, 't10k-labels-idx1-ubyte.gz')
        
        # get training data
        with gzip.open(train_sample_path, 'rb') as sample_path:
            _, train_num, _, _ = struct.unpack('>IIII', sample_path.read(16))
            train_samples = np.frombuffer(sample_path.read(), dtype=np.uint8)
            train_samples = train_samples.reshape(-1, 784)
        with gzip.open(train_label_path, 'rb') as label_path:
            _, train_num = struct.unpack('>II', label_path.read(8))
            train_labels = np.frombuffer(label_path.read(), dtype=np.uint8)
            train_labels = train_labels.reshape(-1, 1) + 1
        
        # divide training data into training set and validation set
        np.random.seed(252)
        index = np.arange(train_num)
        np.random.shuffle(index)
        
        valid_num = 10000
        train_samples, valid_samples = train_samples[index, :][:-valid_num, :], train_samples[index, :][-valid_num:, :]
        train_labels, valid_labels = train_labels[index][:-valid_num], train_labels[index][-valid_num:]
        
        # normalize and store
        train_mean = np.mean(train_samples)
        train_var = np.var(train_samples)
        train_samples = (train_samples - train_mean) / np.sqrt(train_var)
        valid_samples = (valid_samples - train_mean) / np.sqrt(train_var)
        self.train_set = {'sample': train_samples, 'label': train_labels, 'num': train_num-valid_num}
        self.valid_set = {'sample': valid_samples, 'label': valid_labels, 'num': valid_num}
        
        self.mean = train_mean
        self.var = train_var
        
        # get testing data
        with gzip.open(test_sample_path, 'rb') as sample_path:
            _, test_num, _, _ = struct.unpack('>IIII', sample_path.read(16))
            test_samples = np.frombuffer(sample_path.read(), dtype=np.uint8)
            test_samples = test_samples.reshape(-1, 784)
        with gzip.open(test_label_path, 'rb') as label_path:
            _, test_num = struct.unpack('>II', label_path.read(8))
            test_labels = np.frombuffer(label_path.read(), dtype=np.uint8)
            test_labels = test_labels.reshape(-1, 1) + 1
        
        # normalize and store
        test_samples = (test_samples - train_mean) / np.sqrt(train_var)
        self.test_set = {'sample': test_samples, 'label': test_labels, 'num': test_num}
        
        
    def get_random_data(self):
        
        num = self.train_set['num']
        index = np.random.randint(0, num)
        
        sample = self.train_set['sample'][index : (index + 1), :].copy().T
        label = self.train_set['label'][index, 0]
        
        return (sample, label)