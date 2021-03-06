import os
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision

import numpy as np

class DigitsDataset(torch.utils.data.Dataset):
    """
    Loads a digits reordering dataset 
    The additional dict is empty because the raw data is contained in X
    """
    
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.data[idx][0].astype(np.float64)).unsqueeze(-2) # shape (1, n_set)
        Y = torch.from_numpy(self.data[idx][1]) #shape (batch, n_set)
        additionnal_dict = {}
        return X, Y, additionnal_dict
    
    
class WordsDataset(torch.utils.data.Dataset):
    """
    Loads a digits reordering dataset 
    vocab_size = 26 for example if the words were generated using the western alphabet
    """
    
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.data[idx][0].astype(np.float64)) # shape (n_set, max_word_length, vocab_size)
        Y = torch.from_numpy(self.data[idx][1]) # shape (batch, n_set)
        #words = self.data[idx][2] #shape (batch, n_set)
        #additionnal_dict = {'words': words}
        additionnal_dict = {}
        return X, Y, additionnal_dict
    
class VideosDataset(torch.utils.data.Dataset):
    """
    Loads a digits reordering dataset 
    vocab_size = 26 for example if the words were generated using the western alphabet
    """
    
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #X = torch.from_numpy(self.data[idx][0].astype(np.float64)).permute(1,0)  # shape (n_features, n_set) 
        
        #X = {} #need to make X a dict instead of a list because the dataloader would not know how to handle a list
        #for _ in range(len(self.data[idx][0])):
        #    X[_] = torch.from_numpy(self.data[idx][0][_].astype(np.float64))
        
        
        
        sequences = [torch.from_numpy(x) for x in self.data[idx][0]]
        
        #print([x.size(0) for x in sequences])
        
        if max([x.size(0) for x in sequences]) > 1000:
            return None
        
        #print(f'sequences[0] size : {sequences[0].size()}')
        X = pad_sequence(sequences)
        #print(f'X shape: {X.size()}')
        
        Y = torch.from_numpy(self.data[idx][1]) # shape (batch, n_set)
        filename = self.data[idx][2] 
        blocks_boundaries = self.data[idx][3] 
        additionnal_dict = {'filename': filename, 'blocks_boundaries': blocks_boundaries}
        return X, Y, additionnal_dict