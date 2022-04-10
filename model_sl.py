import pickle
import os.path as osp
from copy import deepcopy

def load_model(filename):
    
    print('\nModel loading')
    
    # load the network
    model_path = osp.join('model', filename+'_nn.pkl')
    
    with open(model_path, 'rb') as path:
        network = pickle.load(path)['model']
    
    print('\nComplete.')
    
    return network

def save_model(network, filename):
    
    print('\nModel saving.')
    
    saved_network = deepcopy(network)
    model = {'model': saved_network}
        
    with open(osp.join('model', filename+'_nn.pkl'), 'wb') as f:
        f.truncate()
        pickle.dump(model, f)
        
    print('\nComplete.')
