import os
import torch
import torch.utils.data
from PIL import Image
import torchvision

import numpy as np

class DigitsDataset(torch.utils.data.Dataset):
    """
    Loads a digits reordering dataset 
    """
    
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.data[idx][0].astype(np.float64)).unsqueeze(-2)
        Y = torch.from_numpy(self.data[idx][1])
        return X, Y