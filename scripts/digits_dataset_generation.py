"""
example run: python scripts/digits_dataset_generation.py --n-train 10000 --n-val 2000 --n-test 200 --n-set 5
"""
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
    parser.add_argument('--n-test', type=int, default=200,
                        help='number of validation examples')
    parser.add_argument('--n-set', type=int, default=5,
                        help='size of the set')
    parser.add_argument('--output-folder', type=str, default='./pickles',
                       help='Where to save the generated pickle file')
    
    args = parser.parse_args()
    
    
    X_train = np.random.uniform(size=(args.n_train, args.n_set))
    Y_train = np.argsort(X_train, axis=1)
    ##If you want the output to be a random permutation
    ##Y_train = np.random.permutation(Y_train.transpose()).transpose()

    
    X_val = np.random.uniform(size=(args.n_val, args.n_set))
    Y_val = np.argsort(X_val, axis=1)
    ##If you want the output to be a random permutation
    #Y_val = np.random.permutation(Y_val.transpose()).transpose()

    
    X_test = np.random.uniform(size=(args.n_test, args.n_set))
    Y_test = np.argsort(X_test, axis=1)
    ##If you want the output to be a random permutation
    #Y_test = np.random.permutation(Y_test.transpose()).transpose()
    
    dict_data = {'train': [], 'val': [], 'test': []}
    for i in range(X_train.shape[0]):
        dict_data['train'].append((X_train[i, :], Y_train[i,:]))
    
    for i in range(X_val.shape[0]):
        dict_data['val'].append((X_val[i, :], Y_val[i,:]))
        
    for i in range(X_test.shape[0]):
        dict_data['test'].append((X_test[i, :], Y_test[i,:]))
        
    dt = str(datetime.now()).replace(' ', '_')
    filename = f'digits_reordering_{args.n_train}_{args.n_val}_{args.n_set}_{dt}.pkl'
    
    with open(f'{args.output_folder}/{filename}', 'wb') as f:
        pickle.dump(dict_data, f)
        
        
if __name__ == '__main__':
    main()
    