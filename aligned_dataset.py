import os.path
from base_dataset import BaseDataset, get_transform, normalize
from image_folder import make_dataset
from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    def __init__(self, args, split):
        self.opt = args
        self.root = self.opt.dataroot
        self.phase = split
        #还需要改的 dataroot get_transform

        ### input A (label maps)
        dir_A = '_A'
        self.dir_A = os.path.join(self.root, self.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        dir_B = '_B'
        self.dir_B = os.path.join(self.root, self.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))


        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        transform_A = get_transform(self.opt)
        A_tensor = transform_A(A.convert('RGB'))

        # B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        # if self.phase == 'train' :
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt)
        B_tensor = transform_B(B)

        ### if using instance maps

        # input_dict = {'label_H': B_tensor_H, 'label_W': B_tensor_W, 'label': B_tensor, 'image': A_tensor}
        # input_dict = { 'label': B_tensor_train, 'label_test': B_tensor, 'image': A_tensor}
        # sample['case_name'] = self.A_paths[index].strip((self.root+'/'))

        return A_tensor, B_tensor

    def __len__(self):
        return len(self.A_paths) #// self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'