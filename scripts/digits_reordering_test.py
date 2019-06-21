"""
RUN EXAMPLE: 
- digits: python scripts/digits_reordering_test.py --pickle-file pickles/digits_reordering_10000_2000_5_2019-06-18_13:15:34.234123.pkl --resume checkpoints/5/ep_10_map_inf_latest.pth.tar --hidden-dims 32 --lstm-steps 10

- words: python scripts/digits_reordering_test.py --pickle-file pickles/words_reordering_10000_2000_5_2019-06-18_12:32:55.406161.pkl --resume checkpoints/3/ep_100_map_inf_latest.pth.tar --hidden-dim 32 --lstm-steps 10 --reader words --input-dim 26
"""

# Usual imports
import time
import math
import numpy as np
import os
#import matplotlib.pyplot as plt
import argparse
import pickle
from glob import glob
import random

#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.backends import cudnn
from torch.optim import Adam

#tensorboard
from tensorboardX import SummaryWriter

#my modules
from dataset import DigitsDataset, WordsDataset
from order_matters import ReadProcessWrite
from digits_reordering import create_model

DATASET_CLASSES = {'linear': DigitsDataset, 'words': WordsDataset}
LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def main():
    if torch.cuda.is_available():
        args.USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        args.USE_CUDA = False
        
        
    
    with open(args.pickle_file, 'rb') as f:
        dict_data = pickle.load(f)
        
    
    #runs = glob(args.saveprefix+'/*')
    #it = len(runs) + 1
    #writer = SummaryWriter(os.path.join(args.tensorboard_saveprefix, str(it)))
    #writer.add_text('Metadata', 'Run {} metadata :\n{}'.format(it, args,))
    
    dataset_class = DATASET_CLASSES[args.reader]
    
    test_ds = dataset_class(dict_data['test'])
    
    test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    
    model = create_model(args)
    
    
    
    if args.USE_CUDA:
        device = torch.cuda.current_device()
        #model.cuda()
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        
    test(test_loader, model)
        
def test(test_loader, model):
    
    model.eval()
    
    # Training
    correct_orders = 0
    total_orders = 0
    loader_len = len(test_loader)
    for i, data in enumerate(test_loader, 0):
        X, Y, additional_dict = data
        # Transfer to GPU
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        X, Y = X.to(device).float(), Y.to(device)
        #X, Y = X.cuda().float(), Y.cuda()


        # forward + backward + optimize
        outputs, pointers, hidden = model(X)
        
        outputs = outputs.contiguous().view(-1, outputs.size()[-1])
        #print(f'outputs: {outputs.size()}, Y: {Y.size()}')
        
        
        
        if args.reader == 'words':
            words = X_to_words(X.cpu())
            #inds_x = np.tile(np.array(range(words.shape[0])), [words.shape[1], 1]).T
            predicted_inds = pointers.cpu().data.numpy()
            real_inds = Y.cpu().data.numpy()
            for i in range(real_inds.shape[0]):
                print(f' Predicted Words order: {words[i, predicted_inds[i,:]]}')
                print(f' Real Words order: {words[i, real_inds[i,:]]}\n')
            
        else :
            print(f'Predictions: {pointers}')
            print(f'Real orders: {Y}')
            
        for _ in range(pointers.size(0)):
            total_orders += 1
            if Y[_,:].equal( pointers[_,:]):
                correct_orders +=1
                
    print(f'Fraction of perfectly sorted sets: {correct_orders/total_orders}')


def X_to_words(X):
    """
    X is of shape (batch, n_seq, max_word_length, vocab_size)
    """
    array = X.data.numpy()
    words =  np.ndarray((array.shape[0], array.shape[1]), dtype=object)
    words.fill('')
    #print(f'Words shape: {words.shape}')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                if max(X[i,j,k,:]) == 1:
                    words[i,j] += LETTERS[np.argmax(X[i,j,k,:])]
                else:
                    pass
    return words
                                          
                                        
                                            
    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Music Structure training')
    parser.add_argument('--pickle-file', type=str,
                        help='file from which to load the dictionnary containing the test data info')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint model file (default: none)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='the batch size to use for training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--hidden-dims', type=int, default=[256], nargs='+',
                        help='list of number of hidden dimension for the for each layer of the read block. The last on is also the hidden_dim of the process and write blocks')
    parser.add_argument('--lstm-steps', type=int, default=5,
                        help='number of steps for the self attention process block')
    parser.add_argument('--reader', default='linear', type=str, 
                        help='what reader and dataset class ')
    parser.add_argument('--input-dim', default=1, type=int, 
                        help='dimension of the input ex: 1 for digits, 26 for words create from western alphabet')
    args = parser.parse_args()
    main()
