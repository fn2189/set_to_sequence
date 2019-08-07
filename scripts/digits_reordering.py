"""
RUN EXAMPLE: 
- Digits: python scripts/digits_reordering.py --pickle-file pickles/digits_reordering_10000_2000_10_2019-08-07_12:00:48.139281.pkl  --hidden-dims 32 --lstm-steps 10 --lr 1e-4 --batch-size 32 --epochs 10 --saveprefix checkpoints/digits --tensorboard-saveprefix tensorboard/digits --print-offset 100

- Words: python scripts/digits_reordering.py --pickle-file pickles/words_reordering_10000_2000_10_2019-08-07_15:04:50.672650.pkl  --hidden-dims 32 --lstm-steps 10 --lr 1e-4 --batch-size 256 --epochs 10 --saveprefix checkpoints/words/n_10 --tensorboard-saveprefix tensorboard/words/n_10 --print-offset 100 --reader words --input-dim 26

- Videos: python scripts/digits_reordering.py --pickle-file ../s3-drive/set_to_sequence/resnet50_moments_in_time.pkl  --hidden-dims 256 --lstm-steps 10 --lr 1e-4 --batch-size 128 --epochs 100 --saveprefix checkpoints/videos --tensorboard-saveprefix tensorboard/videos --print-offset 100 --reader videos --input-dim 2048 --dropout .2 --weight-decay 1e-4

- python scripts/digits_reordering.py --pickle-file ../s3-drive/set_to_sequence/video_reordering_resnet50.pkl  --hidden-dims 256 --lstm-steps 10 --lr 1e-4 --batch-size 128 --epochs 100 --saveprefix checkpoints/videos --tensorboard-saveprefix tensorboard/videos --print-offset 100 --reader videos --input-dim 2048 --dropout .2 --weight-decay 1e-4

- Videos (mnv2 + optical flow): python scripts/digits_reordering.py --pickle-file pickles/video_reordering_18374_3937_5_2019-07-17_00:27:53.238932.pkl  --hidden-dim 256 --lstm-steps 10 --lr 1e-4 --batch-size 128 --epochs 200 --saveprefix checkpoints/videos --tensorboard-saveprefix tensorboard/videos --print-offset 25 --reader videos --input-dim 2560 --dropout .2 --weight-decay 1e-4
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
from dataset import DigitsDataset, WordsDataset, VideosDataset
from order_matters import ReadProcessWrite


DATASET_CLASSES = {'linear': DigitsDataset, 'words': WordsDataset, 'videos': VideosDataset}

def main():
    if torch.cuda.is_available():
        args.USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        args.USE_CUDA = False
        
        
    
    with open(args.pickle_file, 'rb') as f:
        dict_data = pickle.load(f)
        
    
    runs = glob(args.saveprefix+'/*')
    it = len(runs) + 1
    writer = SummaryWriter(os.path.join(args.tensorboard_saveprefix, str(it)))
    writer.add_text('Metadata', 'Run {} metadata :\n{}'.format(it, args,))
    
    dataset_class = DATASET_CLASSES[args.reader]
    
    train_ds = dataset_class(dict_data['train'])
    val_ds = dataset_class(dict_data['val'])
    
    train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    model = create_model(args)
    
    args.weights_indices = {}
    args.parameters = list(model.named_parameters())
    for name, param in args.parameters:
        if param.requires_grad:
            size = list(param.data.flatten().size())[0]
            args.weights_indices[name] = random.sample(range(size), min(5, size))
    
    
    if args.USE_CUDA:
        device = torch.cuda.current_device()
        #model.cuda()
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        model.to(device) 
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
        }, is_best, os.path.join(args.saveprefix, str(it), f'ep_{epoch+1}_map_{best_val_loss:.3}'))
    
    writer.close()
    

    
    
def train(train_loader, val_loader, model, criterion, optimizer, epoch, writer):
    
    model.train()
    
    # Training
    running_loss = 0.0
    loader_len = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        X, Y, additional_dict = data
        
        # Transfer to GPU
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        X, Y = X.to(device).float(), Y.to(device)
        #print(f'X shape: {X.size()}, Y shape: {Y.size()}')
        #X, Y = X.cuda().float(), Y.cuda()

        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, pointers, hidden = model(X)
        
        outputs = outputs.contiguous().view(-1, outputs.size()[-1])
        Y = Y.view(-1)
        #print(f'outputs: {outputs.size()}, Y: {Y.size()}')
        
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % args.print_offset == args.print_offset -1:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss /args.print_offset ))
            #print(f'outputs: {outputs[:15,:]}, Y: {Y[:15]}')
            writer.add_scalar('data/losses/train_loss', running_loss/args.print_offset, i + 1 + epoch*loader_len)
            write_weights(args.weights_indices, args.parameters, writer, i + 1 + epoch*loader_len)
            running_loss = 0
    

    # Validation
    avg_val_loss = val(val_loader, model, criterion, epoch)
    writer.add_scalar('data/losses/val_loss', running_loss/args.print_offset, (epoch+1)*loader_len)
    
    return avg_val_loss
    
    
def val(val_loader, model, criterion, epoch=0):

    # switch to eval mode
    model.eval()

    with torch.set_grad_enabled(False):
        val_loss = 0.0
        for cpt, data in enumerate(val_loader, 0):
            X, Y, additional_dict = data

            # Transfer to GPU
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
            X, Y = X.to(device).float(), Y.to(device)
            #X, Y = X.cuda().float(), Y.cuda()

            # forward + backward + optimize
            outputs, pointers, hidden = model(X)

            outputs = outputs.contiguous().view(-1, outputs.size()[-1])
            Y = Y.view(-1)
            loss = criterion(outputs, Y)
            val_loss += loss.item()

    #cpt here is the last cpt in the loop, len(validator_generator) -1
    print(f'Epoch {epoch + 1} validation loss: {val_loss / (cpt+1)}')

    return val_loss / (cpt+1)

def create_model(args):
    print("=> creating model")
    model = ReadProcessWrite(args.hidden_dims, args.lstm_steps, args.batch_size, input_dim= args.input_dim, reader=args.reader)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.USE_CUDA:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            #try:
            #    args.best_map = checkpoint['val_map']
            #except KeyError as e:
            #    args.best_map = None
            # print(checkpoint['state_dict'].keys())
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except RuntimeError as e:
                print('Could not load state_dict. Attempting to correct for DataParallel module.* parameter names. This may not be the problem however...')
                # This catches the case when the model file was save in DataParallel state
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model.load_state_dict(new_state_dict)
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    return model
    

def write_weights(weights_indices, parameters, writer, n_iter):
    """
    Adds a current set of weights to the writer
    
    Parameters
    =========
    weights_indices: dict of the indices of the weights to 
    capture for each flattened weight vector
    
    parameters: list of tuple (name, torch.Tensor parameter vector)
    writer: the tensorboadX writer object
    n_iter: The iteration at which to save
    """
    weights_data = {}
    for name, param in parameters:
        if param.requires_grad:
            indices = weights_indices[name]
            for idx in indices:
                weights_data[f'{idx}'] = param.data.flatten()[idx]
            writer.add_scalars(f'data/weigths/{name}', weights_data, n_iter)
            weights_data = {}
                   

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
    parser.add_argument('--hidden-dims', type=int, default=[256], nargs='+',
                        help='list of number of hidden dimension for the for each layer of the read block. The last on is also the hidden_dim of the process and write blocks')
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
    parser.add_argument('--resume', default='', type=str, 
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--reader', default='linear', type=str, 
                        help='what reader and dataset class ')
    parser.add_argument('--input-dim', default=1, type=int, 
                        help='dimension of the input ex: 1 for digits, 26 for words create from western alphabet')
    parser.add_argument('--dropout', default=0.1, type=float, 
                        help='dropout rate')
    args = parser.parse_args()
    main()