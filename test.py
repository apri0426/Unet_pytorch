from model import Net

import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from aligned_dataset import AlignedDataset
from PIL import Image
import numpy as np
import util
from torch.autograd import Variable
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', help='path to dataset of kaggle ultrasound nerve segmentation')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epoch to start')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')
parser.add_argument('--output_name', default='checkpoint04.pth', type=str, help='output checkpoint filename')

args = parser.parse_args()
# print(args)

############## dataset processing
dataset = AlignedDataset(args, 'test')
# kaggle2016nerve(args.dataroot)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           num_workers=args.workers, shuffle=True)

############## create model
model = Net(args.useBN)
# print(model.state_dict())
if args.cuda:
  model.cuda()
  # cudnn.benchmark = True

############## resume

print("=> loading checkpoint '{}'".format(args.output_name))

checkpoint = torch.load(args.output_name, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint)
# print(model.state_dict())
#  print("=> loaded checkpoint (epoch {}, loss {})"
#      .format(checkpoint['epoch'], checkpoint['loss']) )
# else:
#     print("=> no checkpoint found at '{}'".format(args.resume))



############ just check test (visualization)

def showImg(img, binary=True, fName=''):
  """
  show image from given numpy image
  """
  # img = img[0,0,:,:]

  # if binary:
  #   img = img > 0.5

  img = Image.fromarray(img)

  if fName:
    img.save('assets/'+fName+'.png')
  else:
    img.show()


model.eval()


for i, (x,y) in enumerate(test_loader):
  if i >= 11:
    break
  with torch.no_grad():
    y_pred = model(x)
  empty = torch.empty(160, 480)
  y_pred = y_pred.squeeze()
  print(y_pred.shape)
  # outputs_H = y_pred[0, :, :]
  # outputs_W = y_pred[1, :, :]
  # outputs = torch.stack((outputs_H.squeeze().cuda(), outputs_W.squeeze().cuda(), empty.cuda()), 0)
  showImg(util.tensor2im(x), binary=False, fName='ori_'+str(i))
  showImg(util.tensor2im(outputs), binary=False, fName='pred_'+str(i))
  showImg(util.tensor2label(y), fName='gt_'+str(i))

