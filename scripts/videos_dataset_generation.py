"""
example run: python scripts/videos_dataset_generation.py --glob-str "/MediaArchivePool/datasets/video/Moments_In_Time/Moments_in_Time_Raw/validation/[a-h]*/*" --n-set 5
"""


from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
import os
import sys
import random
import pickle 
import argparse
from datetime import datetime


from PIL import Image
import skvideo.io
import skvideo.datasets
import json
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms


sys.path.append('..')
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
    
    args = parser.parse_args()
    
    videos = glob(args.glob_str)
    
    N_SET = args.n_set
    b_dict = {}
    for i in range(len(videos)):
        metadata = skvideo.io.ffprobe(videos[i])
        length = int(metadata['video']['@nb_frames'])
        #print(videos[i], length)
        boundaries = sorted(random.sample(range(length), N_SET-1)) + [length]    
        for i in range(1, len(boundaries)):
            boundaries[i] = boundaries[i] - boundaries[i-1]
        b_dict[videos[i]] = boundaries
        
        
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
    test_videos = [x for x in not_train if x not in not_train]
    
    
    data_dict = {'train': [], 'test': [], 'val': []}
    for i, vid in enumerate(train_videos):
        X, y = compute_features(test_trans, model, train_videos[i], n_set=N_SET)
        data_dict['train'].append((X, y, vid))

    for i, vid in enumerate(val_videos):
        X, y = compute_features(test_trans, model, val_videos[i], n_set=N_SET)
        data_dict['val'].append((X, y, vid))

    for i, vid in enumerate(test_videos):
        X, y = compute_features(test_trans, model, test_videos[i], n_set=N_SET)
        data_dict['test'].append((X, y, vid))
        
        
    dt = str(datetime.now()).replace(' ', '_')
    filename = f'video_reordering_{args.n_train}_{args.n_val}_{args.n_set}_{dt}.pkl'
    
    with open(f'{args.output_folder}/{filename}', 'wb') as f:
        pickle.dump(dict_data, f)
        
    return



    
def compute_features(transform, model, videofile, n_set=5):
    vid_in = skvideo.io.FFmpegReader(videofile)
    set_vectors = []
    if torch.cuda.is_available():
        model.cuda()
    for i in range(n_set):
        predictions = []
        cpt = 0
        frames_iterator = tqdm(vid_in)
        #for idx, frame in  enumerate(tqdm(vid_in)):
        while cpt < boundaries[i]:
            input_frame = transform(Image.fromarray(frame))
            with torch.no_grad():
                input_img_var = torch.autograd.Variable(input_frame)
            if torch.cuda.is_available():
                input_img_var = input_img_var.cuda()

            # compute output
            output = model(input_img_var.unsqueeze(0))
            # this assumes that the model has two outputs 
            # ['emb'] and ['prob'] and we want the first output
            if torch.cuda.is_available():
                output = output[0].cpu().data.numpy()

            # check that output has same dimensionality as expected labels
            #assert output.shape[1] == len(all_classes)
            predictions.append(output)
            cpt += 1

        vidout = np.array(predictions) 
        vidout = vidout.mean(axis=0)
        #print(vidout.shape)
        set_vectors.append(vidout)
        vid_in.close()
        
    #This is the random order in which we shuffle the "blocks" of the video
    #So we need to figure out the inverse permutation function that is going to serve as the correct order
    random_order = random.sample(range(n_set), n_set)
    y = np.zeros(n_set, dtype=int)
    print(random_order, len(set_vectors))
    for k, v in enumerate(random_order):
        y[v] = k
    # we reorder the feature representation of the blocks now that we have computed the the correct order from the 
    #shuffled one
    set_vectors = [set_vectors[x] for x in random_order]
    X = np.stack(set_vectors, axis=0)
        
    return X, y
    
    
if __name__ == '__main__':
    main()