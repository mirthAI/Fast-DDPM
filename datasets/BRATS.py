import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import torch

from .sr_util import get_paths_from_npys, brats_transform_augment


class BRATS(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        self.img_size = img_size
        self.data_len = data_len
        self.split = split
        img_root = dataroot + '/A/'
        gt_root = dataroot + '/B/'
        self.img_npy_path, self.gt_npy_path = get_paths_from_npys(img_root, gt_root)
        self.data_len = len(self.img_npy_path)
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_FD = None
        img_LD = None
        base_name = None
        extension = None
        number = None
        FW_path = None
        BW_path = None

        base_name = self.img_npy_path[index].split('/')[-1]
        case_name = base_name.split('.')[0]
        
        img_npy = np.load(self.img_npy_path[index])
        img = Image.fromarray(img_npy)
        gt_npy = np.load(self.gt_npy_path[index])
        gt = Image.fromarray(gt_npy)
        img = img.resize((self.img_size, self.img_size))
        gt = gt.resize((self.img_size, self.img_size))

        [img, gt] = brats_transform_augment(
            [img, gt], split=self.split)

        return {'FD': gt, 'LD': img, 'case_name': case_name}






