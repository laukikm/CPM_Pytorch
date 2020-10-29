"""


"""
from data_loader.dataLoader import CarDataset as Mydata
from model.cpm import CPM
from src.util import heatmap_image,save_images,PCK

import os
import torch
import torch.optim as optim
import torch.nn as nn
import configparser

from torch.autograd import Variable
from torch.utils.data import DataLoader

online=False
config = configparser.ConfigParser()
device_ids = [0]  
# *********************** hyper parameter  ***********************
if not online:
    # multi-GPU
    config.read('conf.text')

if online:
    config.read('conf_online.text')

    train_data_dir = config.get('data', 'train_data_dir')
    model_save_dir=config.get('data', 'model_save_dir')
    
    save_dir = config.get('data', 'save_dir')
    
    learning_rate = config.getfloat('training', 'learning_rate')
    batch_size = config.getint('training', 'batch_size')
    epochs = config.getint('training', 'epochs')
    begin_epoch = config.getint('training', 'begin_epoch')
    n_keypoints=config.getint('training', 'n_keypoints')
    
cuda = torch.cuda.is_available()
 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# *********************** Build dataset ***********************
train_data = Mydata(data_dir=train_data_dir)


print ('Train dataset total number of images sequence is ----' + str(len(train_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# *********************** Build model ***********************

net = CPM(out_c=n_keypoints)

if cuda:
    net = net.cuda(device_ids[0])
    net = nn.DataParallel(net, device_ids=device_ids)


if begin_epoch > 0:
    save_path = model_save_dir + str(begin_epoch) + '.pth'
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)


def train():
    # *********************** initialize optimizer ***********************
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss(size_average=True)                       # loss function MSE average

    net.train()
    for epoch in range(begin_epoch, epochs + 1):
        print ('epoch....................' + str(epoch))
        
        for step, (image, label_map, center_map, imgs) in enumerate(train_dataset):
            
            image = Variable(image.cuda() if cuda else image)                   # 4D Tensor
            # Batch_size  *  3  *  width(368)  *  height(368)

            # 4D Tensor to 5D Tensor
            label_map = torch.stack([label_map]*6, dim=1)
            # Batch_size  *  21 *   45  *  45
            # Batch_size  *   6 *   21  *  45  *  45
            label_map = Variable(label_map.cuda() if cuda else label_map)

            center_map = Variable(center_map.cuda() if cuda else center_map)    # 4D Tensor
            # Batch_size  *  width(368) * height(368)

            optimizer.zero_grad()
            pred_6 = net(image, center_map)  # 5D tensor:  batch size * stages * 21 * 45 * 45 #This is huge

            # ******************** calculate loss of each joints ********************
            loss = criterion(pred_6, label_map)

            # backward
            loss.backward()
            loss=loss.detach()
            #import ipdb
            #ipdb.set_trace()
            
            optimizer.step() #Not using SGD
            #gc.collect()

            if step % 10 == 0:
                print ('--step .....' + str(step))
                print ('--loss ',loss.item()*10000)
                gb_factor=10**9
                print('--gpu memory=',torch.cuda.memory_allocated()/gb_factor)

            #if step % 200 == 0:
            #    save_images(label_map[:, 5, :, :, :], pred_6[:, 5, :, :, :], step, epoch, imgs)

        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'model_epoch{:d}.pth'.format(epoch)))
        
            #Delete tensors from memory
            #del image,label_map,center_map,pred_6

    print ('train done!')
    
if __name__ == '__main__':
    train()








