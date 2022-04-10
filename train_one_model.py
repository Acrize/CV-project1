import argparse
import numpy as np
from layer import fc_layer, activate_layer, softmax_layer
from model_sl import save_model
from data_loader import data_loader
from network import fc_network

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_one_model')
    
    parser.add_argument('--model_name', action='store', type=str)
    parser.add_argument('--max_iter', action='store', type=int)
    parser.add_argument('--learning_rate', action='store', type=float)
    parser.add_argument('--penalty', action='store', type=float)
    parser.add_argument('--neural_num', action='store', type=int)
    parser.add_argument('--visualize', action='store', type=str)
    
    args = parser.parse_args()
    
    max_iter = args.max_iter
    learning_rate = args.learning_rate
    penalty = args.penalty
    neural_num = args.neural_num
    
    layers = [
        fc_layer(in_dim=784, out_dim=neural_num, sigma=np.sqrt(2/(784+neural_num))),
        activate_layer(),
        fc_layer(in_dim=neural_num, out_dim=10, sigma=np.sqrt(2/(10+neural_num))),
        softmax_layer()]
    
    nn = fc_network(layers)
    
    dataset = data_loader()
    
    error = nn.train(max_iter=max_iter,
                    learning_rate=learning_rate,
                    data=dataset,
                    valid=False, 
                    penalty=penalty)
    
    save_model(nn, args.model_name)
    
    if args.visualize == 'y':
        nn.visualize_error()
        nn.visualize_loss()
        nn.visualize_weight()
