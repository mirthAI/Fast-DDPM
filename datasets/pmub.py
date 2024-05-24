from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import random
import matplotlib.pyplot as plt
from .sr_util import get_valid_paths_from_images, get_valid_paths_from_test_images, transform_augment


class PMUB(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        self.img_size = img_size
        self.data_len = data_len
        self.split = split

        self.img_path = get_valid_paths_from_images(dataroot)
        self.test_img_path = get_valid_paths_from_test_images(dataroot)

        if self.split == 'test':
            self.dataset_len = len(self.test_img_path)
        else:
            self.dataset_len = len(self.img_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_FW = None
        img_MD = None
        img_BW = None
        base_name = None
        extension = None
        number = None
        FW_path = None
        BW_path = None

        base_name = self.img_path[index].split('_')[0]
        case_name = int(base_name.split('/')[-1].split('-')[-1])
        extension = self.img_path[index].split('_')[-1].split('.')[-1]
        number = int(self.img_path[index].split('_')[-1].split('.')[0])
        FW_path = base_name + '_' + str(number+1) + '.' + extension
        BW_path = base_name + '_' + str(number-1) + '.' + extension

        img_BW = Image.open(BW_path).convert("L")
        img_MD = Image.open(self.img_path[index]).convert("L")
        img_FW = Image.open(FW_path).convert("L")
        
        img_BW = img_BW.resize((self.img_size, self.img_size))
        img_MD = img_MD.resize((self.img_size, self.img_size))
        img_FW = img_FW.resize((self.img_size, self.img_size))

        [img_BW, img_MD, img_FW] = transform_augment(
            [img_BW, img_MD, img_FW], split=self.split, min_max=(-1, 1))

        return {'BW': img_BW, 'MD': img_MD, 'FW': img_FW, 'Index': index, 'case_name': case_name}
