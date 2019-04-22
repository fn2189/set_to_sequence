import numpy as np
import pickle 
import argparse
from datetime import datetime


def main():
    
    parser = argparse.ArgumentParser(description='CNN Music Structure training')
    parser.add_argument('--n-train', type=int, default=10000,
                        help='number of training examples ')
    parser.add_argument('--n-val', type=int, default=2000,
                        help='number of validation examples')
    parser.add_argument('--n-set', type=int, default=5,
                        help='size of the set')
    parser.add_argument('--output-folder', type=str, default='./pickles',
                       help='Where to save the generated pickle file')
    
    args = parser.parse_args()
    
    
    X_train = np.random.uniform(size=(args.n_train, args.n_set))
    y_list_indices_train = np.argsort(X_train, axis=1)
    Y_train = np.zeros((args.n_train, args.n_set, args.n_set))
    for i in range(args.n_train):
        Y_train[i, range(args.n_set), y_list_indices_train[i,:]] = 1
    
    X_val = np.random.uniform(size=(args.n_val, args.n_set)
    y_list_indices_val = np.argsort(X_val, axis=1)
    Y_val = np.zeros((args.n_val, args.n_set, args.n_set))
    for i in range(args.n_val):
        Y_val[i, range(args.n_set), y_list_indices_val[i,:]] = 1
    
    dict_data = {'train': [], 'val': []}
    for i in range(X_train.shape[0]):
        dict_data['train'].append((X_train[i, :], Y_train[i,:]))
    
    for i in range(X_val.shape[0]):
        dict_data['val'].append((X_val[i, :], Y_val[i,:]))
        
    dt = str(datetime.now()).replace(' ', '_')
    filename = f'digits_reordering_{args.n_train}_{args.n_val}_{args.n_set}_{dt}.pkl'
    
    with open(f'{args.output_folder}/{filename}', 'wb') as f:
        pickle.dump(dict_data, f)
        
        
if __name__ == '__main__':
    main()
    