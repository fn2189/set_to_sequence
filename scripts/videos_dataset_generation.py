"""
example run: python scripts/videos_dataset_generation.py --glob-str "/home/ubuntu/s3-drive/RLY/RLYMedia/*" --n-set 5 --batch-size 64

python scripts/videos_dataset_generation.py --glob-str "/MediaArchivePool/datasets/video/Moments_In_Time/Moments_in_Time_Raw/validation/[a-h]*/*" --n-set 5 --batch-size 128
"""


from glob import glob
import numpy as np
import pandas as pd
import os
import sys
import random
import pickle 
import argparse
from datetime import datetime
import pdb


from PIL import Image
import skvideo.io
import skvideo.datasets
import json
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms


sys.path.append('.')
from scripts.mobilenet import mnv2, pretrained_model_fnames



def main():
    
    parser = argparse.ArgumentParser(description='CNN Music Structure training')
    parser.add_argument('--glob-str', type=str,
                        default='/MediaArchivePool/datasets/video/Moments_In_Time/Moments_in_Time_Raw/validation/[a-h]*/*',
                        help='the glob string to pass to get the list of videos')
    parser.add_argument('--n-set', type=int, default=5,
                        help='size of the set')
    parser.add_argument('--output-folder', type=str, default='./pickles',
                       help='Where to save the generated pickle file')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='the maximum number of frames whose feature can be computed in a single batch. This is to e adapted to hardware capabilities')
    
    args = parser.parse_args()
    
    videos = glob(args.glob_str)
    
    N_SET = args.n_set
    """
    b_dict = {}
    for i in range(len(videos)):
        metadata = skvideo.io.ffprobe(videos[i])
        length = int(metadata['video']['@nb_frames'])
        #print(videos[i], length)
        boundaries = sorted(random.sample(range(length), N_SET-1)) + [length]    
        for i in range(1, len(boundaries)):
            boundaries[i] = boundaries[i] - boundaries[i-1]
        b_dict[videos[i]] = boundaries
     """   
        
    # This normalize is specific to ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    input_size = 224
    test_trans = transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
        ])
    
    #TO DO: expose other kwargs for the model through command line args
    model = mnv2(pretrained=True, freeze=False) #look at ../scripts/mobilenet.py to see additional args
    
    n_videos = len(videos)
    n_train = int(n_videos*.7)
    n_val = int(n_videos*.15)

    train_videos = random.sample(videos, n_train)
    not_train = [x for x in videos if x not in train_videos]
    val_videos = random.sample(not_train, n_val)
    test_videos = [x for x in not_train if x not in val_videos]
    
    
    data_dict = {'train': [], 'test': [], 'val': []}
    #Entering debugger
    #pdb.set_trace()
    for i, vid in enumerate(train_videos):
        #print(f'{i}: {vid}')
        X, y, B = compute_features(test_trans, model, train_videos[i], n_set=N_SET, batch_size=args.batch_size)
        # We need to account for the videos that are to short, that we discard
        if X is None:
            continue
        data_dict['train'].append((X, y, vid, B))
        if (i+1) % 100 == 0:
            print(f'Computed features for video {i}: {train_videos[i]}')
    print('Training set done')
    
    for i, vid in enumerate(val_videos):
        X, y, B = compute_features(test_trans, model, val_videos[i], n_set=N_SET, batch_size=args.batch_size)
        # We need to account for the videos that are to short, that we discard
        if X is None:
            continue
        data_dict['val'].append((X, y, vid, B))
        if (i+1) % 100 == 0:
            print(f'Computed features for video {i}: {val_videos[i]}')
    print('Val set done')

    for i, vid in enumerate(test_videos):
        X, y, B = compute_features(test_trans, model, test_videos[i], n_set=N_SET, batch_size=args.batch_size)
        # We need to account for the videos that are to short, that we discard
        if X is None:
            continue
        data_dict['test'].append((X, y, vid, B))
        if (i+1) % 100 == 0:
            print(f'Computed features for video {i}: {test_videos[i]}')
            
    print('Test set done')
        
        
    dt = str(datetime.now()).replace(' ', '_')
    filename = f'video_reordering_{n_train}_{n_val}_{args.n_set}_{dt}.pkl'
    
    with open(f'{args.output_folder}/{filename}', 'wb') as f:
        pickle.dump(data_dict, f)
        
    return



    
def compute_features(transform, model, videofile, n_set=5, batch_size=64):
    vid_in = skvideo.io.FFmpegReader(videofile)
    (length, _, _, _) = vid_in.getShape() # numFrame x H x W x channels
    
    if length < 30:
        return None, None, None

    n_frames_per_block = [0]*n_set
    ## We don't want a segment to be less than 2 frames so we keep regenerating boundaries
    ## Until that condition is met
    while min(n_frames_per_block) < 2: 
        #print(f'length: {length}')
        boundaries = [0] + sorted(random.sample(range(length-1), n_set-1)) + [length-1]    
        n_frames_per_block = [boundaries[i] - boundaries[i-1] for i in range(1, len(boundaries))]
        
    #print(n_frames_per_block)
        
    set_vectors = []
    frames_iterator = vid_in.nextFrame()
    if torch.cuda.is_available():
        model.cuda() 
        
    """
        
    global_count= 0
    for i in range(n_set):
        predictions = []
        quotient = n_frames_per_block[i] // batch_size ##how many chunks of size batch_size in n_frames for the i-th block of the video
        n_chunks = quotient if n_frames_per_block[i] % batch_size == 0 else quotient +1 
        
        #print(f'n_frames_per_block[i]: {n_frames_per_block[i]}')
        outputs = []
        for it1 in range(n_chunks):
            # because the number of frames in a segment could be bigger than the max batch size the hardware can support
            # I use an inner while loop to splits those frames in chunks of size args.batch_size
            cpt = 0
            input_frames_array = []
            while (cpt < batch_size ) and ((it1*batch_size +cpt) < n_frames_per_block[i]) :
                #print(cpt, it1)
                cpt += 1
                global_count +=1
                
                try:
                    frame = next(frames_iterator)
                except:
                    print(f'videofile: {videofile}')
                    print(f'total n frames: {length}, i: {i}, n_frames_per_block: {n_frames_per_block}, it1*batch_size +cpt: {it1*batch_size +cpt}')
                    print(f'global_count: {global_count}')
                    raise ValueError('StopIteration')
                input_frame = transform(Image.fromarray(frame))
                input_frames_array.append(input_frame)
                
                
                #assert (cpt <= batch_size )
                #assert ((it1*batch_size +cpt) <= n_frames_per_block[i])
                #assert global_count <= length
                

            

            try:
                input_frames = torch.stack(input_frames_array, dim=0)
            except:
                print(f'it1: {it1}, cpt: {cpt}, n_frames_per_block[i]: {n_frames_per_block[i]}')
                raise ValueError('RuntimeError: expected a non-empty list of Tensors')

            with torch.no_grad():
                input_imgs_var = torch.autograd.Variable(input_frames)
            if torch.cuda.is_available():
                input_imgs_var = input_imgs_var.cuda()

            # compute output
            try:
                output = model(input_imgs_var)
            except:
                print(f'input_imgs_var size: {input_imgs_var.size()}')
                raise ValueError('RuntimeError: CUDA error: out of memory')

            if torch.cuda.is_available():
                output = output.cpu().data.numpy()

            outputs.append(output)
        
        #print(f'global_count: {global_count}')            
            
        vidout = outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)  
        vidout = vidout.mean(axis=0)
        #print(vidout.shape)
        set_vectors.append(vidout)
    
    """
    cpt=0
    input_frames_array = []
    outputs = []
    for frame in frames_iterator:
        input_frame = transform(Image.fromarray(frame))
        input_frames_array.append(input_frame)
        cpt+=1
        
        if cpt % batch_size == 0:
            input_frames = torch.stack(input_frames_array, dim=0)
            with torch.no_grad():
                input_imgs_var = torch.autograd.Variable(input_frames)
            if torch.cuda.is_available():
                input_imgs_var = input_imgs_var.cuda()

            # compute output
            try:
                output = model(input_imgs_var)
            except:
                print(f'input_imgs_var size: {input_imgs_var.size()}')
                raise ValueError('RuntimeError: CUDA error: out of memory')

            if torch.cuda.is_available():
                output = output.cpu().data.numpy()

            outputs.append(output)
            input_frames_array = []
            
    ##Computing the last batch at the end of the loop
    if len(input_frames_array) > 0:
        input_frames = torch.stack(input_frames_array, dim=0)
        with torch.no_grad():
            input_imgs_var = torch.autograd.Variable(input_frames)
        if torch.cuda.is_available():
            input_imgs_var = input_imgs_var.cuda()

        # compute output
        try:
            output = model(input_imgs_var)
        except:
            print(f'input_imgs_var size: {input_imgs_var.size()}')
            raise ValueError('RuntimeError: CUDA error: out of memory')

        if torch.cuda.is_available():
            output = output.cpu().data.numpy()

        outputs.append(output)
        input_frames_array = []
            
    vidout = outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
            
            
            
    ## Now we use the randomly sampled boundaries to define the video segments segments
    segments = [vidout[boundaries[i-1]:boundaries[i]] for i in range(1,len(boundaries))]
    print(f'len(segments): {len(segments)}')
    
    ### We average the features of the frames belonging to each segment to compute the features for the segment
    set_vectors =[x.mean(axis=0) for x in segments]
    
        
        
    #Closing the video stream
    vid_in.close()
        
    #This is the random order in which we shuffle the "blocks" of the video
    #So we need to figure out the inverse permutation function that is going to serve as the correct order
    random_order = random.sample(range(n_set), n_set)
    y = np.zeros(n_set, dtype=int)
    #print(random_order, len(set_vectors))
    for k, v in enumerate(random_order):
        y[v] = k
    # we reorder the feature representation of the blocks now that we have computed the the correct order from the 
    #shuffled one
    set_vectors = [set_vectors[x] for x in random_order]
    X = np.stack(set_vectors, axis=0)
        
    return X, y, boundaries
    
    
if __name__ == '__main__':
    main()