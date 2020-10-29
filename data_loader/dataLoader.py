"""
This is the data loader of the annotated car data

There are four variables in one item
image   Type: Tensor    Size: 3 * 900 * 900 (to be resized to 368,368)
label   Type: Tensor    Size: 18 * 45 * 45
center  Type: Tensor    Size: 3 * 368 * 368 (same as the image)
name    Type:  str

The data is organized in the following style

----data                        This is the folder name like train or test
--------Image1               
-------- .....
--------ImageFinal                   This is organized linearly

----label                        This is the folder name like train or test
--------001L0.json               This is one sequence of images
--------001L1.json
------------ ....

To have a better understanding, you can view ../dataset in this repo
"""

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import PIL as pil
import scipy.misc

import sys
sys.path.append("../")
from src.util import *

import glob

def human_sort_key(name_string):
    num_string=''
    for char in name_string:
        if(ord(char)>=48 and ord(char)<=57): #Only numeric numbers added
            num_string+=char
    return int(num_string)
class CarDataset(Dataset):

    def __init__(self, data_dir, n_keypoints=18, transform=None, sigma=1):
        self.height = 368
        self.width = 368
        

        self.images_dir=sorted(glob.glob(data_dir+"Images/*.PNG"),key=human_sort_key) #It's okay if they aren't in the human order
        self.label_dir = sorted(glob.glob(data_dir+"Labels/*.txt"),key=human_sort_key)# But they need to be in the same order

        self.transform = transform
        self.n_keypoints = n_keypoints  # 21 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        #self.gen_imgs_dir()

    def gen_imgs_dir(self):
        """
        get directory of all images
        :return:
        """

        for seq in self.seqs:               # 001L0
            if seq == '.DS_Store':
                continue
            image_path = os.path.join(self.data_dir, seq)  #
            imgs = os.listdir(image_path)  # [0005.jpg, 0011.jpg......]
            for i in range(len(imgs)):
                self.images_dir.append(image_path + '/' + imgs[i])  #

        print ('total numbers of image is ' + str(len(self.images_dir)))

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        images          3D Tensor      3                *   height(368)      *   weight(368)
        label_map       3D Tensor      (n_keypoints + 1)     *   label_size(45)   *   label_size(45)
        center_map      3D Tensor      1                *   height(368)      *   weight(368)
        """

        label_size = int(self.width / 8) - 1         # 45
        img = self.images_dir[idx]              # '.../001L0/L0005.jpg'

        file=self.label_dir[idx]
        labels = pd.read_csv(file,delimiter=" ")

        # get image
        im = Image.open(img)                # read image
        w, h, c = np.asarray(im).shape      # weight 256 * height 256 * 3
        ratio_x = self.width / float(w)
        ratio_y = self.height / float(h)    # 368 / 256 = 1.4375
        im = im.resize((self.width, self.height))                       # unit8      weight 368 * height 368 * 3
        image = transforms.ToTensor()(im)   # 3D Tensor  3 * height 368 * weight 368

        # get label map
        
        label = np.array([labels['x'],labels['y'] ])       # 0005  list       21 * 2
        label=np.transpose(label)   #n_keypointsx2 array
        
        lbl = self.genLabelMap(label, label_size=label_size, n_keypoints=self.n_keypoints, ratio_x=ratio_x, ratio_y=ratio_y)
        label_maps = torch.from_numpy(lbl)

        # generate the Gaussian heat map
        center_map = self.genCenterMap(x=self.width / 2.0, y=self.height / 2.0, sigma=21,
                                       size_w=self.width, size_h=self.height)
        center_map = torch.from_numpy(center_map)

        return image.float(), label_maps.float(), center_map.float(), img

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def genLabelMap(self, label, label_size, n_keypoints, ratio_x, ratio_y):
        """
        generate label heat map
        :param label:               list            n_keypoints * 2
        :param label_size:          int             45
        :param n_keypoints:              int             21
        :param ratio_x:             float           1.4375
        :param ratio_y:             float           1.4375
        :return:  heatmap           numpy           n_keypoints * boxsize/stride * boxsize/stride
        """
        # initialize
        label_maps = np.zeros((n_keypoints, label_size, label_size))
        background = np.zeros((label_size, label_size))

        # each joint
        for i in range(n_keypoints):
            lbl = label[i]                      # [x, y]
            x = lbl[0] * ratio_x / 8.0          # modify the label
            y = lbl[1] * ratio_y / 8.0
            heatmap = self.genCenterMap(y, x, sigma=self.sigma, size_w=label_size, size_h=label_size)  # numpy
            background += heatmap               # numpy
            label_maps[i, :, :] = np.transpose(heatmap)  # !!!

        return label_maps  # numpy           label_size * label_size * (n_keypoints + 1)


# test case
if __name__ == "__main__":
    root_dir="../../../Final Data/mustang/"
    data = CarDataset(data_dir=root_dir)
    
    img, label, center, name = data[1]
    print ('dataset info ... ')
    print (img.shape)         # 3D Tensor 3 * 368 * 368
    print (label.shape)       # 3D Tensor 21 * 45 * 45
    print (center.shape )     # 2D Tensor 368 * 368
    print (name)              # str   ../dataset/train_data/001L0/L0461.jpg

    # ***************** draw label map *****************
    print ('draw label map ....')
    lab = np.asarray(label)
    out_labels = np.zeros(((45, 45)))
    for i in range(18):
        out_labels += lab[i, :, :]
    scipy.misc.imsave('img/car_label.jpg', out_labels)

    # ***************** draw image *****************
    print ('draw heat map ....')
    im_size = 368
    img = transforms.ToPILImage()(img)
    img.save('img/car_img.jpg')
    heatmap = np.asarray(label[0, :, :])

    im = Image.open('img/car_img.jpg')

    heatmap_image(img, lab,n_keypoints=18, save_dir='img/car_heat.jpg')
    







