import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self):
        pass



def get_transform(opt, method=Image.BICUBIC, normalize=True):
    transform_list = []


    # if opt.resize_or_crop == 'none':
    #     base = float(2 ** opt.n_downsample_global)
    #     if opt.netG == 'local':
    #         base *= (2 ** opt.n_local_enhancers)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, 16, method)))
    #
    # if opt.isTrain and not opt.no_flip:
    #     transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    # transform_list.append(transforms.Pad(padding=[0, 4], fill=0, padding_mode='constant'))
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)



def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
