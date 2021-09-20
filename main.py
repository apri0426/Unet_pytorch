from model import Net

import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from aligned_dataset import AlignedDataset
from PIL import Image
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
parser.add_argument('--output_name', default='checkpoint.pth', type=str, help='output checkpoint filename')

args = parser.parse_args()
print(args)

############## dataset processing
dataset = AlignedDataset(args, 'train')
# kaggle2016nerve(args.dataroot)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=True)

############## create model
model = Net(args.useBN)
if args.cuda:
  model.cuda()
  # cudnn.benchmark = True

############## resume
if args.resume:
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))

    if args.cuda == False:
      checkpoint = torch.load(args.resume, map_location={'cuda:0':'cpu'})

    args.start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, loss {})"
        .format(checkpoint['epoch'], checkpoint['loss']) )
  else:
    print("=> no checkpoint found at '{}'".format(args.resume))

#
# def save_checkpoint(state, filename=args.output_name):
#   torch.save(state, filename)

############## training
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)
model.train()

def train(epoch):
  """
  training
  """
  loss_fn = nn.MSELoss()
  if args.cuda:
    loss_fn = loss_fn.cuda()

  loss_sum = 0

  for i, (x, y) in enumerate(train_loader):
    x, y_true = x, y
    if args.cuda:
      x = x.cuda()
      y_true = y_true.cuda()

    for ii in range(1):
      y_pred = model(x)

      loss = loss_fn(y_pred, y_true)
      
      optimizer.zero_grad()
      loss.backward()
      loss_sum += loss.item()

      optimizer.step()
 
    if i % 5 == 0:
      print('batch no.: {}, loss: {}'.format(i, loss.item()))

  print('epoch: {}, epoch loss: {}'.format(epoch,loss.item()/len(train_loader) ))

  # save_checkpoint({
  #   'epoch': epoch + 1,
  #   'state_dict': model.state_dict(),
  #   'loss': loss.item()/len(train_loader)
  # })

for epoch in range(args.niter):
  train(epoch)
  if epoch >= args.niter - 1:
    torch.save(model.state_dict(), args.output_name)

# ############ just check test (visualization)

def showImg(img, binary=True, fName=''):
  """
  show image from given numpy image
  """
  # img = img[0,0,:,:]
  #
  # if binary:
  #   img = img > 0.5

  img = Image.fromarray(img)

  if fName:
    img.save('assets/'+fName+'.png')
  else:
    img.show()


model.eval()
dataset = AlignedDataset(args, 'test')
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           num_workers=args.workers, shuffle=True)

for i, (x,y) in enumerate(test_loader):
  if i >= 11:
    break

  with torch.no_grad():
    y_pred = model(x.cuda())
  showImg(util.tensor2im(x), binary=False, fName='ori_'+str(i))
  showImg(util.tensor2im(y_pred), binary=False, fName='pred_'+str(i))
  showImg(util.tensor2label(y), fName='gt_'+str(i))

