from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
from .sr_util import get_paths_from_images, get_valid_paths_from_images, get_valid_paths_from_test_images, transform_augment


class LDFDCT(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        self.img_size = img_size
        self.data_len = data_len
        self.split = split
        self.img_ld_path, self.img_fd_path = get_paths_from_images(dataroot)
        self.data_len = len(self.img_ld_path)

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
        
        base_name = self.img_ld_path[index].split('/')[-1]
        case_name = base_name.split('_')[0]
        
        img_LD = Image.open(self.img_ld_path[index]).convert("L")
        img_FD = Image.open(self.img_fd_path[index]).convert("L")
        img_LD = img_LD.resize((self.img_size, self.img_size))
        img_FD = img_FD.resize((self.img_size, self.img_size))

        [img_LD, img_FD] = transform_augment(
            [img_LD, img_FD], split=self.split, min_max=(-1, 1))

        return {'FD': img_FD, 'LD': img_LD, 'case_name': case_name}
