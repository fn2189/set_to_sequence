# Usual imports
import time
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
from glob import glob

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
from dataset import DigitsDataset
from order_matters import ReadProcessWrite


def main():
    if torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False
        
    
    with open(args.pickle_file, 'rb') as f:
        dict_data = pickle.load(f)
        
    
    runs = glob(args.saveprefix+'/*')
    it = len(runs) + 1
    writer = SummaryWriter(os.path.join(args.tensorboard_saveprefix, str(it)))
    writer.add_text('Metadata', 'Run {} metadata :\n{}'.format(it, args,))
    
    train_ds = DigitsDataset(dict_data['train'])
    val_ds = DigitsDataset(dict_data['val'])
    
    train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    input_dim = dict_data['train'][0][0].shape[0]
    model = ReadProcessWrite(args.hidden_dim, args.lstm_steps, args.batch_size, input_dim)
    
    
    if USE_CUDA:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=args.lr)
    
    best_val_loss = np.inf
    for ind, epoch in enumerate(range(args.epochs)):
        val_loss = train(train_loader, val_loader, model, criterion, optimizer, epoch, writer)

        
        is_best = val_loss > best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, is_best, os.path.join(cmd_args.saveprefix, str(it), f'ep_{epoch+1}_map_{best_val_loss:.3}'))
    
    writer.close()
    

    
    
def train(train_loader, val_loader, model, criterion, optimizer, epoch, writer):
    
    model.train()
    
    # Training
    running_loss = 0.0
    loader_len = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        X, Y = data
        
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        X, Y = X.cuda().float(), Y.cuda()

        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, pointers, hidden = model(X)
        
        outputs = outputs.contiguous().view(-1, o.size()[-1])
        Y = Y.view(-1)
        
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % args.print_offset == args.print_offset -1:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss /args.print_offset ))
            writer.add_scalar('data/losses/train_loss', running_loss/cmd_args.print_offset, i + 1 + epoch*loader_len)
            running_loss = 0
    

    # Validation
    avg_val_loss = val(val_loader, model, criterion, epoch)
    writer.add_scalar('data/losses/test_loss', running_loss/cmd_args.print_offset, (epoch+1)*loader_len)
    
    
def val(val_loader, model, criterion, epoch=0):

    # switch to eval mode
    model.eval()

    with torch.set_grad_enabled(False):
        val_loss = 0.0
        for cpt, data in enumerate(val_loader, 0):
            X, Y = data

            # Transfer to GPU
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            X, Y = X.cuda().float(), Y.cuda()
            
            # Model computations
            # forward + backward + optimize
            outputs, pointers, hidden = model(X)
            
            outputs = outputs.contiguous().view(-1, o.size()[-1])
            Y = Y.view(-1)
            loss = criterion(outputs, Y)
            val_loss += loss.item()

    #cpt here is the last cpt in the loop, len(validator_generator) -1
    print(f'Epoch {epoch + 1} validation loss: {val_loss / (cpt+1)}')

    return val_loss / (cpt+1)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Music Structure training')
    parser.add_argument('--pickle-file', type=str,
                        help='file from which to load the dictionnary containing the training data info')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='number of hidden dimension for the reader/processor/writer')
    parser.add_argument('--lstm-steps', type=int, default=5,
                        help='number of steps for the self attention process block')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='Nesterov')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size to use for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs for training')
    parser.add_argument('--saveprefix', type=str,
                        help='folder where to save the checkpoint files')
    parser.add_argument('--tensorboard-saveprefix', type=str,
                        help='folder where to save the tensorboardX  files')
    parser.add_argument('--print-offset', type=int, default=10,
                        help='how often to print in minibatches')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    main()