"""
example run: python scripts/words_dataset_generation.py --n-train 10000 --n-val 2000 --n-test 200 --n-set 10 --max-length 30 --min-length 5 --from-list True
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
    parser.add_argument('--from-list', type=bool, default=False,
                       help='whether or not to pull words from an english dictionnary or not')
    
    
    args = parser.parse_args()
    
    train_list = None
    val_list = None
    test_list = None
    
    words_list = load_words()
    
    
    if args.from_list:
        """
        n_words = len(words_list)
        n_train = int(n_words*.7)
        n_val = int(n_words*.15)
        
        train_list = random.sample(words_list, n_train)
        not_train = [x for x in words_list if x not in train_list]
        val_list = random.sample(not_train, n_val)
        test_list = [x for x in not_train if x not in val_list]
        """
        
        n_train = args.n_train
        n_val = args.n_val
        n_test = args.n_test
        n_set = args.n_set
        
        words = random.sample(words_list, n_set*(n_train+ n_val+ n_test)) ##N_samples*n_set
        
        train_list = words[:n_set*n_train]
        val_list = words[n_set*n_train:n_set*(n_train+n_val)]
        test_list = words[n_set*(n_train+n_val):]
        

        
    
        
    
    X_train, Y_train, W_train = generate_set(args.n_train, args.n_set, args.max_length, args.min_length, w_list=train_list)

    X_val, Y_val, W_val = generate_set(args.n_val, args.n_set, args.max_length, args.min_length, w_list=val_list)
    
    X_test, Y_test, W_test = generate_set(args.n_test, args.n_set, args.max_length, args.min_length, w_list=test_list)
    
    
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



def load_words():
    with open('../english-words/words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
        
    return valid_words



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

def generate_set(N, n_set, max_length, min_length, w_list=None):
    l_word = []
    l_enc = []
    #if w_list:
    #    N = len(w_list)
    print('train set length: ', N)
    for i in range(N):
        l_sub_word = []
        l_sub_enc = []
        for j in range(n_set):
            if w_list:
                encoding, word = generate_word_from_list(i*n_set+j, w_list, max_length)
            else:
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

def generate_word_from_list(i, w_list, max_length):
    LETTERS = 'abcdefghijklmnopqrstuvwxyz'
    LETTERS_DICT = {}
    for _, letter in enumerate(LETTERS):
        LETTERS_DICT[letter] = _
    
    try:
        word = w_list[i]
    except:
        raise ValueError(f'{i} out of index for w_list')
    encoding = np.zeros((max_length, len(LETTERS)))
    for _ in range(len(word)):
        letter = word[_]
        encoding[_, LETTERS_DICT[letter]] = 1
        
    return encoding, word
        
if __name__ == '__main__':
    main()