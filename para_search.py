from copy import deepcopy
import numpy as np
from layer import fc_layer, activate_layer, softmax_layer
from network import fc_network
from data_loader import data_loader
import os.path as osp
from model_sl import load_model, save_model

if __name__ == '__main__':
    
    with open(osp.join('model', 'search_process.txt'), 'w') as f:
        f.truncate()
    
    dataset = data_loader()
    
    max_iter = 300000
    learning_rate_space = [5e-3, 2e-3, 1e-3]
    penalty_space = [0.01, 0.001]
    neural_num_space = [50, 100, 300]
    
    lowest_error = 1.0
    best_paras = None
    
    for learning_rate in learning_rate_space:
        for penalty in penalty_space:
            for neural_num in neural_num_space:
                
                nn = fc_network(
                    layers = [
                        fc_layer(in_dim=784, out_dim=neural_num, sigma=np.sqrt(2/(784+neural_num))),
                         activate_layer(),
                        fc_layer(in_dim=neural_num, out_dim=10, sigma=np.sqrt(2/(10+neural_num))),
                        softmax_layer()])  
                
                with open(osp.join('model', 'search_process.txt'), 'a') as f:
                    f.write('\n')
                    f.write('Parameters: ' + str((learning_rate, penalty, neural_num)))
                    f.write('\n')
                
                print('\n')
                print('Parameters: ' + str((learning_rate, penalty, neural_num)))
                
                error = nn.train(max_iter=max_iter,
                                learning_rate=learning_rate,
                                data=dataset,
                                valid=True, 
                                penalty=penalty)
                
                with open(osp.join('model', 'search_process.txt'), 'a') as f:
                    f.write('\n')
                    f.write('One model finish. The error of validation is %5f' % error)
                    f.write('\n')
                
                print('\n')
                print('One model finish. The error of validation is %5f' % error)
                
                if error < lowest_error:
                    lowest_error = error
                    best_paras = (learning_rate, penalty, neural_num)
                    best_network = deepcopy(nn)
                    
                    with open(osp.join('model', 'search_process.txt'), 'a') as f:
                        f.write('\n')
                        f.write('Parameters update.')
                        f.write('\n')
                        
                    print('\n')
                    print('Parameters update.')
                    
    save_model(best_network, 'best')
    
    with open(osp.join('model', 'search_process.txt'), 'a') as f:
        f.write('\n')
        f.write('Best Parameters:')
        f.write('\n')
        f.write(str(best_paras))
        f.write('\n')


    nn_load = load_model('best')
    
    nn_load.visualize_error()
    nn_load.visualize_loss()
    nn_load.visualize_weight()