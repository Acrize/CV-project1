import argparse
from model_sl import load_model
from data_loader import data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    
    parser.add_argument('--model_name', action='store', type=str)
    
    args = parser.parse_args()
    
    network = load_model(args.model_name)
    
    dataset = data_loader()
    
    network.test(dataset)