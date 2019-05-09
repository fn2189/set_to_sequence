# MobileNet v1 adapted from https://raw.githubusercontent.com/marvis/pytorch-mobilenet/
# MobileNet v2 adapted from https://github.com/tonylins/pytorch-mobilenet-v2
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


__all__ = ['MobileNetV1', 'MobileNetV2', 'mnv1', 'mnv2']


pretrained_model_fnames = { 
    'mnv1': '../data/mnv1/mobilenet_sgd_rmsprop_69.526.tar',
    'mnv2': '../data/mnv2/mobilenet_v2.pth.tar',
}



class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.features = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.last_channel = 1024
        self.cls =nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, num_classes),
        )

        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.cls(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    
### MN v2 ###
def conv_bn(inp, oup, stride, export=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        (nn.ReLU(inplace=True) if export else nn.ReLU6(inplace=True)) # Used instead of ReLU6 for ONNX -> CoreML conversion
    )


def conv_1x1_bn(inp, oup, export=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        (nn.ReLU(inplace=True) if export else nn.ReLU6(inplace=True)) # Used instead of ReLU6 for ONNX -> CoreML conversion        
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, export=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                (nn.ReLU(inplace=True) if export else nn.ReLU6(inplace=True)), # Used instead of ReLU6 for ONNX -> CoreML conversion        
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                (nn.ReLU(inplace=True) if export else nn.ReLU6(inplace=True)), # Used instead of ReLU6 for ONNX -> CoreML conversion        
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                (nn.ReLU(inplace=True) if export else nn.ReLU6(inplace=True)), # Used instead of ReLU6 for ONNX -> CoreML conversion        
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1., export=False):
        '''
        <export> argument should be set to True when instatiating this
        model for export to coreml format. Removes Dropout layer.
        '''
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, export=export)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, export=export))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, export=export))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, export=export))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building cls
        if export:
            #self.cls = nn.Sequential(
            self.classifier = nn.Sequential(
                nn.Linear(self.last_channel, num_classes, bias = True),
            )
        else:
            #self.cls = nn.Sequential(
            self.classifier = nn.Sequential(
                # Dropout is not supported by onnx-coreml export
                nn.Dropout(),                
                nn.Linear(self.last_channel, num_classes, bias = True),
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        #x = self.cls(x)
        #x = self.classifier(x)
        return x
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     n = m.weight.size(1)
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()


                

def mnv1(pretrained=False, **kwargs):
    """Constructs a MobileNet version 1 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV1(**kwargs)
    if pretrained:
        if pretrained == 'True' or pretrained == True:
            pretrained = pretrained_model_fnames['mnv1']
        print(f'Pretrained fname {pretrained}')
        load_pretrained_weights(pretrained, model)
    return model 

def mnv2(pretrained=False, freeze=False, **kwargs):
    """Constructs a MobileNet version 2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        if pretrained == 'True' or pretrained == True:
            pretrained = pretrained_model_fnames['mnv2']
        print(f'Pretrained fname {pretrained}')
        load_pretrained_weights(pretrained, model)
    if freeze:
        # freeze_layers(pretrained, model)
        subnet = model.features
        freeze_subnet(subnet, layer_limit = 15) # TODO: TESTING THIS !!!!
        
        # Check that not all layers got frozen
        assert model.cls[0].weight.requires_grad == True
        assert model.cls[0].bias.requires_grad == True
    return model 


from collections import OrderedDict
import os

def load_pretrained_weights(pretrained_model_fname, model):
    '''
    This function is generic to model architecture
    '''
    if not os.path.isfile(pretrained_model_fname):
        print(f'Error ******* Weights file does not exist {pretrained_model_fname}')
        return False
    if torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_model_fname)
    else:
        pretrained_dict = torch.load(pretrained_model_fname, map_location='cpu')

    # check if this is a checkpoint or a modelfile
    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k 
        if k not in model.state_dict().keys():
            # If the pretrained model was saved as DataParallel and
            # the current model is not
            if k[:7] == 'module.':                
                name = k[7:] # remove `module.`
        if model.state_dict()[name].shape == pretrained_dict[k].shape:
            new_state_dict[name] = v
        else:
            print(f'Using Kaiming He normal initialization for {name} {model.state_dict()[name].shape}')
            try:
                new_state_dict[name] = nn.init.kaiming_normal_(model.state_dict()[name])
            except ValueError as e:
                print(f'0 val init for {name} {model.state_dict()[name].shape}')
                new_state_dict[name] = nn.init.constant_(model.state_dict()[name], 0)
    # load params
    model.load_state_dict(new_state_dict)

# def freeze_layers(pretrained_model_fname, model):
#     '''
#     Freeze layers loaded from pretrained model (generic to architecture)
#     '''
#     pretrained_dict = torch.load(pretrained_model_fname)
#     if 'state_dict' in  pretrained_dict.keys():
#         pretrained_dict = pretrained_dict['state_dict']
#     print(f' ****** FREEEZING LAYERS!!!!!! \n {pretrained_dict.keys()} *****')
#     freeze(model, list(pretrained_dict.keys()))

# def freeze(module, layers_to_freeze, prefix=''):
#     for name, child in module._modules.items():
#         if child is not None:
#             for pname, param in child.named_parameters():
#                 if prefix + name + '.' + pname in layers_to_freeze:
#                     param.requires_grad = False
#             freeze(child, layers_to_freeze, prefix + name + '.')

def freeze_subnet(module, layer_limit = 1e6):
    for name, child in module._modules.items():
        # Stops freezing layers above layer limit This presumes
        # sequential structure with layer blocks that have increasing
        # integer key values in the module map
        try:
            if int(name) > layer_limit:
                return
        except ValueError as e:
            pass
        print(f'************************** Currently in freeze_subnet and looking at {name} {child}')
        if child is not None:
            for pname, param in child.named_parameters():
                param.requires_grad = False
            freeze_subnet(child, layer_limit)
