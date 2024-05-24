import os
import torch
import torchvision
import random
import numpy as np
import glob

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def extract_number(filename):
    number = int(filename.split('_')[1].split('.')[0])
    return number

# LDFDCT
def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

    ld_images = glob.glob(path + "**/**/*ld.png", recursive=True)
    fd_images = glob.glob(path + "**/**/*fd.png", recursive=True)

    assert ld_images, '{:s} has no valid ld image file'.format(path)
    assert fd_images, '{:s} has no valid fd image file'.format(path)
    assert len(ld_images) == len(fd_images), 'Low Dose images nd Full Dose images are not paired!'
    return sorted(ld_images), sorted(fd_images)

# Single SR
def get_paths_from_single_sr_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

    lr_images = glob.glob(path + "**/**/*lr.png", recursive=True)
    hr_images = glob.glob(path + "**/**/*hr.png", recursive=True)

    assert lr_images, '{:s} has no valid lr image file'.format(path)
    assert hr_images, '{:s} has no valid hr image file'.format(path)
    assert len(lr_images) == len(hr_images), 'Low Dose images nd Full Dose images are not paired!'
    return sorted(lr_images), sorted(hr_images)


def get_paths_from_npys(path_data, path_gt):
    assert os.path.isdir(path_data), '{:s} is not a valid directory'.format(path_data)
    assert os.path.isdir(path_gt), '{:s} is not a valid directory'.format(path_gt)

    data_npy = glob.glob(path_data + "*.npy")
    gt_npy = glob.glob(path_gt + "*.npy")

    assert data_npy, '{:s} has no valid data npy file'.format(path_data)
    assert gt_npy, '{:s} has no valid GT npy file'.format(path_gt)
    assert len(data_npy) == len(gt_npy), 'Low Dose images nd Full Dose images are not paired!'
    return sorted(data_npy), sorted(gt_npy)


# Delete head and tail for train and val
def get_valid_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []

    for dirpath, folder_path, fnames in sorted(os.walk(path)):
        
        filtered_fnames = [fname for fname in fnames if fname.endswith('.png') and not fname.startswith('.')]
        fnames = filtered_fnames

        fnames = sorted(fnames, key=extract_number)
        new_fnames = fnames[1:-1]

        for fname in new_fnames:
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)

    assert images, '{:s} has no valid image file'.format(path)
    return images


# Delete tail for test
def get_valid_paths_from_test_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []

    for dirpath, _, fnames in sorted(os.walk(path)):
        filtered_fnames = [fname for fname in fnames if not fname.startswith('.')]
        fnames = filtered_fnames

        fnames = sorted(fnames, key=extract_number)
        new_fnames = fnames[:-1]

        for fname in new_fnames:
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
                
    assert images, '{:s} has no valid image file'.format(path)
    return images


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
Resize = torchvision.transforms.Resize((224, 224), antialias=True)
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)

    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


def brats_transform_augment(img_list, split='val'):
    imgs = [totensor(img) for img in img_list]
    # imgs = [Resize(img) for img in imgs_tlist]
    # if split == 'train':
    #     imgs = torch.stack(imgs, 0)
    #     imgs = hflip(imgs)
    #     imgs = torch.unbind(imgs, dim=0)
    ret_img = [img.clamp(-1., 1.) for img in imgs]
        
    return ret_img
