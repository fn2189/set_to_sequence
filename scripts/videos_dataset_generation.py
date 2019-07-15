"""
example run: python scripts/videos_dataset_generation.py --glob-str "/home/ubuntu/s3-drive/RLY/RLYMedia/*.mp4" --n-set 5 --batch-size 256 --with-flow

python scripts/videos_dataset_generation.py --glob-str "/MediaArchivePool/datasets/video/Moments_In_Time/Moments_in_Time_Raw/validation/[a-h]*/*" --n-set 5 --batch-size 64 
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
import math
from scipy.misc import imread, imresize


from PIL import Image
import skvideo.io
import skvideo.datasets
import json
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms


sys.path.append('.')
sys.path.append('./flownet2_pytorch')
from scripts.mobilenet import mnv2, pretrained_model_fnames
from flownet2_pytorch.models import FlowNet2



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
    parser.add_argument('--with-flow', action='store_true',
                       help='whether to use optical flow or raw images')
    parser.add_argument('--fp16', action='store_true', help='Run flow_model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
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
    if torch.cuda.is_available():
        model.cuda() 
        
    flow_model = None
    if args.with_flow:
        flow_model = FlowNet2(args)
        path = '../s3-drive/flownet/FlowNet2_checkpoint.pth.tar'
        pretrained_dict = torch.load(path)['state_dict']
        model_dict = flow_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        flow_model.load_state_dict(model_dict)
        flow_model.cuda()
    
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
        X, y, B = compute_features_main(test_trans, model, train_videos[i], n_set=N_SET, batch_size=args.batch_size, flow_model=flow_model)
        # We need to account for the videos that are to short, that we discard
        if X is None:
            continue
        data_dict['train'].append((X, y, vid, B))
        if (i+1) % 100 == 0:
            print(f'Computed features for video {i}: {train_videos[i]}')
    print('Training set done')
    
    for i, vid in enumerate(val_videos):
        X, y, B = compute_features_main(test_trans, model, val_videos[i], n_set=N_SET, batch_size=args.batch_size, flow_model=flow_model)
        # We need to account for the videos that are to short, that we discard
        if X is None:
            continue
        data_dict['val'].append((X, y, vid, B))
        if (i+1) % 100 == 0:
            print(f'Computed features for video {i}: {val_videos[i]}')
    print('Val set done')

    for i, vid in enumerate(test_videos):
        X, y, B = compute_features_main(test_trans, model, test_videos[i], n_set=N_SET, batch_size=args.batch_size, flow_model=flow_model)
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
    
    
    
    """
    ### Since there seems to be sporadic I/O bugs, I retry 10 times so I loop for 10 times (for the retries and I break if success (if 
    ###len(outputs) > 0
    outputs = []
    
    #try:
    vid_in = skvideo.io.FFmpegReader(videofile)
    #except:
    #    print(f'error in video reading for video file: {videofile}')
    #    return None, None, None
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


    cpt=0
    input_frames_array = []

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
            #try:
            output = model(input_imgs_var)
            #except:
            #    print(f'input_imgs_var size: {input_imgs_var.size()}')
            #    raise ValueError('RuntimeError: CUDA error: out of memory')

            if torch.cuda.is_available():
                output = output.cpu()

            output = output.data.numpy()
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
        #try:

        output = model(input_imgs_var)
        #except:
        #    print(f'input_imgs_var size: {input_imgs_var.size()}')
        #    raise ValueError('RuntimeError: CUDA error: out of memory')

        if torch.cuda.is_available():
            output = output.cpu()

        output = output.data.numpy()
        outputs.append(output)
        input_frames_array = []

    try:
        vidout = outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
    except:
        print(f'ouputs empty for video file: {videofile}')
        raise ValueError('AssertionError')
        #return None, None, None
            
            
            
    ## Now we use the randomly sampled boundaries to define the video segments segments
    segments = [vidout[boundaries[i-1]:boundaries[i]] for i in range(1,len(boundaries))]
    #print(f'len(segments): {len(segments)}')
    
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
    """
    #pdb.set_trace()
    video = skvideo.io.vread(videofile)
    length = video.shape[0]
    
    if length < 30:
        return None, None, None

    n_frames_per_block = [0]*n_set
    ## We don't want a segment to be less than 2 frames so we keep regenerating boundaries
    ## Until that condition is met
    while min(n_frames_per_block) < 2: 
        #print(f'length: {length}')
        boundaries = [0] + sorted(random.sample(range(length-1), n_set-1)) + [length-1]    
        n_frames_per_block = [boundaries[i] - boundaries[i-1] for i in range(1, len(boundaries))]
        
    
    #set_vectors = []
    
        
    

    runs = math.ceil((length/batch_size))
    outputs = []

    for i in range(runs):
        """
        input_frames = torch.from_numpy(video[j*batch_size:min((j+1)*batch_size, video.shape[0]-1),:,:,:]).permute(0,3,1,2)
        input_frames = input_frames.type(torch.FloatTensor)
        with torch.no_grad():
            input_imgs_var = torch.autograd.Variable(input_frames)
        if torch.cuda.is_available():
            input_imgs_var = input_imgs_var.cuda()
        
        print(f'Size: {input_imgs_var.size()}')
        """
        input_frames_array = []
        for j in range(i*batch_size,min((i+1)*batch_size, video.shape[0])):
            frame = video[j,:,:,:]
            input_frame = transform(Image.fromarray(frame))
            input_frames_array.append(input_frame)
            
        input_frames = torch.stack(input_frames_array, dim=0)
        with torch.no_grad():
            input_imgs_var = torch.autograd.Variable(input_frames)
        if torch.cuda.is_available():
            input_imgs_var = input_imgs_var.cuda()
        # compute output
        #try:
        output = model(input_imgs_var)
        #except:
        #    print(f'input_imgs_var size: {input_imgs_var.size()}')
        #    raise ValueError('RuntimeError: CUDA error: out of memory')

        if torch.cuda.is_available():
            output = output.cpu()

        output = output.data.numpy()
        outputs.append(output)
            
    vidout = outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
                     
    ## Now we use the randomly sampled boundaries to define the video segments segments
    segments = [vidout[boundaries[i-1]:boundaries[i]] for i in range(1,len(boundaries))]
    #print(f'len(segments): {len(segments)}')
    
    ### We average the features of the frames belonging to each segment to compute the features for the segment
    set_vectors =[x.mean(axis=0) for x in segments]
        
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
            
def compute_features_2(transform, image_model, flow_model, videofile, n_set=5, batch_size=64):
    
    video = skvideo.io.vread(videofile)
    
    ##need to resize all the video to the same shape
    resized_frames = []
    for _ in range(video.shape[0]):
        resized_frames.append(imresize(video[_,:,:,:], (256,256))) ## Look for a better way and shape for the resize operation 
    resized_video = np.stack(resized_frames)
    img_pairs = np.stack([resized_video[:resized_video.shape[0]-1], resized_video[1:]]).transpose(1,4,0,2,3)
    
    
    
    length = video.shape[0] - 1 ##because we lose an index y considering consecutive pairs of images
    
    if length < 30:
        return None, None, None

    n_frames_per_block = [0]*n_set
    ## We don't want a segment to be less than 2 frames so we keep regenerating boundaries
    ## Until that condition is met
    while min(n_frames_per_block) < 5: 
        #print(f'length: {length}')
        boundaries = [0] + sorted(random.sample(range(length-1), n_set-1)) + [length-1]    
        n_frames_per_block = [boundaries[i] - boundaries[i-1] for i in range(1, len(boundaries))]
        
    runs = math.ceil((length/batch_size))
    outputs = []

    for i in range(runs):
        
        """
        for j in range(i*batch_size,min((i+1)*batch_size, img_pairs.shape[0])):
            frame = video[j,:,:,:]
            input_frame = transform(Image.fromarray(frame))
            input_frames_array.append(input_frame)
            
        input_frames = torch.stack(input_frames_array, dim=0)
        """
        inputs = torch.from_numpy(img_pairs[i*batch_size:min((i+1)*batch_size, img_pairs.shape[0]),:,:,:,:])
        #try:
            
        inputs = inputs.float()
        #except:
        #    print(f'inputs: {inputs}')
        
        #print(f'inputs type: {inputs.type()}')    
        
        with torch.no_grad():
            inputs_var = torch.autograd.Variable(inputs)
        if torch.cuda.is_available():
            inputs_var = inputs_var.cuda()
        # compute the output of the flow model
        #
        output1 = flow_model(inputs_var)
        if torch.cuda.is_available():
            output1 = output1.cpu()
        output1 = output1.permute(0,2,3,1)

        output1 = output1.data.numpy()
        
        ##pass the output of the flow model to the image model to compute features for each flow image
        input_frames_array = []
        for j in range(len(output1)):
            frame = output1[j,:,:,:]
            frame = np.concatenate([frame, np.zeros((frame.shape[0], frame.shape[1], 1))], axis=-1) ##we add a 3rd channel to each flow to make it an image
            frame = np.uint8(frame*255/np.max(frame)) #rescaling and converting to int to transform into PIL image
            #print(f'frame : {frame.dtype}')
            input_frame = transform(Image.fromarray(frame))
            input_frames_array.append(input_frame)
            
        input_frames = torch.stack(input_frames_array, dim=0)
        with torch.no_grad():
            input_imgs_var = torch.autograd.Variable(input_frames)
        if torch.cuda.is_available():
            input_imgs_var = input_imgs_var.cuda()
        # compute output
        #try:
        output = image_model(input_imgs_var)
        #except:
        #    print(f'input_imgs_var size: {input_imgs_var.size()}')
        #    raise ValueError('RuntimeError: CUDA error: out of memory')

        if torch.cuda.is_available():
            output = output.cpu()

        output = output.data.numpy()
        outputs.append(output)
        
    vidout = outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
                     
    ## Now we use the randomly sampled boundaries to define the video segments segments
    ##We remove the last optical flow in each segment because it contains information about the following segment
    segments = [vidout[boundaries[i-1]:boundaries[i]-1] for i in range(1,len(boundaries))]
    #print(f'len(segments): {len(segments)}')
    
    ### We average the features of the frames belonging to each segment to compute the features for the segment
    set_vectors =[x.mean(axis=0) for x in segments]
        
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

def compute_features_main(transform, model, videofile, n_set=5, batch_size=64, flow_model=None):
    if flow_model:
        return compute_features_2(transform, model, flow_model, videofile, n_set=n_set, batch_size=batch_size)
    else:
        return compute_features(transform, model, videofile, n_set=n_set, batch_size=batch_size)
    
    
if __name__ == '__main__':
    main()
