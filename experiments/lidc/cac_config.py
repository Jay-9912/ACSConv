import time
import os
import sys

class CACEnv():  # need to modify
    data = '/cluster/home/it_stu167/wwj/adrenal/' # The cac data location
    video_resnet18_pretrain_path = '.../resnet-18-kinetics.pth' 
    mednet_resnet18_pretrain_path = '.../MedicalNet_pytorch_files/pretrain/resnet_18_23dataset.pth' 

class CACSegConfig(): 
    batch_size = 8
    n_epochs = 120
    drop_rate = 0.0
    seed = 0
    num_workers = 8

    # optimizer
    lr = 0.001
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    gamma=0.1

    save_all = True
    use_syncbn = True

    backbone = 'resnet18' # resnet18, densenet121, vgg16
    conv = 'ACSConv' # ACSConv, Conv2_5d, Conv3d

    # modify this if conv is not Conv3d
    pretrained = False # True, False 
    # modify this if conv is Conv3d, only resnet18 is able to load video / mednet weights
    pretrained_3d = 'i3d' # i3d, video, mednet, nopretrained 

    if conv=='Conv3d':
        if pretrained_3d in ['video', 'mednet'] and backbone != 'resnet18':
            raise ValueError('Only resnet18 is able to load video / mednet weights')
        pretrained = False
        flag = '_' + pretrained_3d
    else:
        flag = '_pretrained' if pretrained else '_nopretrained'
    save = os.path.join('/cluster/home/it_stu167/wwj/classification_after_crop/result/CACSeg', backbone, conv, time.strftime("%y%m%d_%H%M%S")+flag)

class CACClassConfig(CACSegConfig):   # need to modify
    batch_size = 24
    n_epochs = 160
    drop_rate = 0.0
    seed = 0
    num_workers = 8

    # optimizer
    lr = 0.001
    max_lr= 0.01
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    # milestones = []
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    gamma=0.1

    save_all = True
    use_syncbn = True

    backbone = 'resnet18' # resnet18, densenet121, vgg16
    conv = 'ACSConv' # ACSConv, Conv2_5d, Conv3d
    pretrained = True # True, False

    flag = '_pretrained' if pretrained else '_nopretrained'
    save = os.path.join('/cluster/home/it_stu167/wwj/classification_after_crop/result/CACClass', backbone, conv, time.strftime("%y%m%d_%H%M%S")+flag) # need to modify
