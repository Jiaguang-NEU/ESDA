import os
import os.path
import copy
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm
from .get_class import get_img_class

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2,
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0], target_pix[1]] = 1
                        if tmp_label.sum() >= 2 * 32 * 32:
                            new_label_class.append(c)
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list



class SemData(Dataset):
    def __init__(self, split=3, data_root=None, data_list=None, transform=None, mode='train', use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.split = split
        self.data_root = data_root
        self.use_coco = use_coco

        if not use_coco:
            self.class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3:
                self.sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81))
                    self.sub_val_list = list(range(1, 21))

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        self.transform = transform

        data_list_split = (data_list.split('/')[0:-1])
        data_list_path = copy.deepcopy(data_list_split)
        data_list_path.append('{0}/data_list_{1}.txt'.format(self.mode, self.split))
        data_list_path = '/'.join(data_list_path)
        print(data_list_path)

        sub_class_file_list_path = copy.deepcopy(data_list_split)
        sub_class_file_list_path.append('{0}/sub_class_file_list_{1}.txt'.format(self.mode, self.split))
        sub_class_file_list_path = '/'.join(sub_class_file_list_path)
        print(sub_class_file_list_path)

        # # Write FSS Data
        # if self.mode == 'train':
        #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list, filter_intersection=True)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        # elif self.mode == 'val':
        #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list, filter_intersection=False)
        #     # 
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)
        #
        # os.makedirs(os.path.dirname(data_list_path), exist_ok=True)
        # with open(data_list_path, 'w') as f:
        #     for item in self.data_list:
        #         img, label = item
        #         f.write(img + ' ')
        #         f.write(label + '\n')
        # with open(sub_class_file_list_path, 'w') as f:
        #     f.write(str(self.sub_class_file_list))

        # Read FSS Data
        with open(data_list_path, 'r') as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            img, mask = line.split(' ')
            self.data_list.append((img, mask.strip()))

        with open(sub_class_file_list_path, 'r') as f:
            f_str = f.read()
        self.sub_class_file_list = eval(f_str)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0
        if len(label_class) >= 3:
            with open('/home/data/ljg/zw/PFECLIP-main' + '/mutil-label_image.txt', 'a+') as f:
                f.write(image_path + ' ')
                f.write(label_path + '\n')



        class_chosen = label_class[random.randint(1,len(label_class))-1]
        # class_chosen = 6
        img_cls = get_img_class(class_chosen, self.use_coco)
        subcls_list = []
        if self.mode == 'train':
            subcls_list.append(self.sub_list.index(class_chosen))  # list.index(A) return the index of A in the list
            if class_chosen not in self.sub_list:
                print('the dataset processing is not a ZSS processing')
        else:
            subcls_list.append(self.sub_val_list.index(class_chosen))
            if class_chosen not in self.sub_val_list:
                print('the dataset processing is not a ZSS processing')
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = 255
        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.mode == 'train':
            return image, label, img_cls, subcls_list
        else:
            return image, label, raw_label, img_cls, subcls_list







