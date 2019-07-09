import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.misc import imread, imresize
from skimage.transform import resize
import torch
from torch.autograd import Variable

import argparse

#from FlowNet2_src import FlowNet2
from models import FlowNet2
#from FlowNet2_src import flow_to_image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()
    
    
    
    # Prepare img pair
    #im1 = imread('FlowNet2_src/example/0img0.ppm')
    im1 = imresize(imread('../s3-drive/abbey/images/sun_aacphuqehdodwawg.jpg'), (256,256))
    #im2 = imread('FlowNet2_src/example/0img1.ppm')
    im2 = imresize(imread('../s3-drive/abbey/images/sun_aakbdcgfpksytcwj.jpg'), (256,256))
    # B x 3(RGB) x 2(pair) x H x W
    #ims = np.array([[im1, im2]])
    print(f'images shapes: {im1.shape}, {im2.shape}')
    ims = np.expand_dims(np.stack([im1, im2]), 0)
    print('ims array shape: ', ims.shape)
    ims = ims.transpose((0, 4, 1, 2, 3)).astype(np.float32)
    ims = torch.from_numpy(ims)
    print(ims.size())
    ims_v = Variable(ims.cuda(), requires_grad=False)

    # Build model
    flownet2 = FlowNet2(args)
    path = '../s3-drive/flownet/FlowNet2_checkpoint.pth.tar'
    pretrained_dict = torch.load(path)['state_dict']
    model_dict = flownet2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2.load_state_dict(model_dict)
    flownet2.cuda()

    pred_flow = flownet2(ims_v).cpu().data
    pred_flow = pred_flow[0].numpy().transpose((1,2,0))
    print('pred_flow shape', pred_flow.shape)
    #flow_im = flow_to_image(pred_flow)

    # Visualization
    #plt.imshow(flow_im)
    #plt.savefig('flow.png', bbox_inches='tight')