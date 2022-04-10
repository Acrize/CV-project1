import argparse
from model_sl import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize')
    
    parser.add_argument('--model_name', action='store', type=str)
    
    args = parser.parse_args()
    
    network = load_model(args.model_name)
    
    network.visualize_error()
    network.visualize_loss()
    network.visualize_weight()