"""
example run: python scripts/words_dataset_generation.py --n-train 10000 --n-val 2000 --n-test 200 --n-set 5 --max-length 25 --min-length 5
"""
import numpy as np
import pickle 
import argparse
from datetime import datetime
import random


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
    parser.add_argument('--max-length', type=int, default=25,
                        help='maximum length of the generated words')
    parser.add_argument('--min-length', type=int, default=5,
                        help='minimum length of the generated words')
    parser.add_argument('--output-folder', type=str, default='./pickles',
                       help='Where to save the generated pickle file')
    
    args = parser.parse_args()
    
    
    X_train, Y_train, W_train = generate_set(args.n_train, args.n_set, args.max_length, args.min_length)

    X_val, Y_val, W_val = generate_set(args.n_val, args.n_set, args.max_length, args.min_length)
    
    X_test, Y_test, W_test = generate_set(args.n_test, args.n_set, args.max_length, args.min_length)
    
    
    dict_data = {'train': [], 'val': [], 'test': []}
    for i in range(X_train.shape[0]):
        dict_data['train'].append((X_train[i, :, :, :], Y_train[i,:], W_train[i, :]))
    
    for i in range(X_val.shape[0]):
        dict_data['val'].append((X_val[i, :, :, :], Y_val[i,:], W_val[i, :]))
        
    for i in range(X_test.shape[0]):
        dict_data['test'].append((X_test[i, :, :, :], Y_test[i,:], W_test[i, :]))
        
    dt = str(datetime.now()).replace(' ', '_')
    filename = f'words_reordering_{args.n_train}_{args.n_val}_{args.n_set}_{dt}.pkl'
    
    print(f'Saving pickle to {filename}')
    
    with open(f'{args.output_folder}/{filename}', 'wb') as f:
        pickle.dump(dict_data, f)
        
    return
   

def generate_word(max_length, min_length):
    LETTERS = 'abcdefghijklmnopqrstuvwxyz'
    LETTERS_DICT = {}
    for i, letter in enumerate(LETTERS):
        LETTERS_DICT[letter] = i
    word_length = random.choice(range(min_length, max_length))
    word = ''
    encoding = np.zeros((max_length, len(LETTERS)))
    for i in range(word_length):
        letter = random.choice(LETTERS)
        word = word + letter
        encoding[i, LETTERS_DICT[letter]] = 1
        
                   
    return encoding, word

def generate_set(N, n_set, max_length, min_length):
    l_word = []
    l_enc = []
    for i in range(N):
        l_sub_word = []
        l_sub_enc = []
        for j in range(n_set):
            encoding, word = generate_word(max_length, min_length)
            l_sub_word.append(word)
            l_sub_enc.append(encoding)
                 
        set_enc = np.stack(l_sub_enc, axis=0)
        l_enc.append(set_enc)
        l_word.append(l_sub_word)
        
    X = np.stack(l_enc, axis=0) #shape (N, n_set, max_length, 26)
    #print(X.shape)
    words_array = np.array(l_word) #shape (N, n_set), array or words, usefu;l to generate Y by argsort
    Y = np.argsort(words_array, axis=1)
    return X, Y, words_array
        
        
if __name__ == '__main__':
    main()