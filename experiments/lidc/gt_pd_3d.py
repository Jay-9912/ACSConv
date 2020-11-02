import sys
sys.path.append('/cluster/home/it_stu167/wwj/classification_after_crop/ACSConv/experiments/')
from mylib.plot_3d import plotly_3d_scan_to_html
import numpy as np
import _init_paths
import fire
import time
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from mylib.sync_batchnorm import DataParallelWithCallback
from cac_dataset_tta3 import CACSegDataset
from mylib.utils import MultiAverageMeter, save_model, log_results, to_var, set_seed, \
        to_device, initialize, categorical_to_one_hot, copy_file_backup, redirect_stdout, \
        model_to_syncbn
from mylib.metrics import cal_batch_iou, cal_batch_dice
from mylib.loss import soft_dice_loss

from cac_config import CACSegConfig as cfg
from cac_config import CACEnv as env

from resnet import FCNResNet
from densenet import FCNDenseNet
from vgg import FCNVGG
from unet import UNet
from acsconv.converters import ACSConverter, SoftACSConverter, Conv3dConverter, Conv2_5dConverter
from load_pretrained_weights_funcs import load_mednet_pretrained_weights, load_video_pretrained_weights


def main(save_path=cfg.save, 
         n_epochs=cfg.n_epochs, 
         seed=cfg.seed
         ):
    # set seed
    if seed is not None:
        set_seed(cfg.seed)
    cudnn.benchmark = True
    # back up your code
    os.makedirs(save_path)
    copy_file_backup(save_path)
    redirect_stdout(save_path)

    # Datasets
    test_set = CACSegDataset(crop_size=[48,48,48], data_path=env.data, random=cfg.random, datatype=2)

    # Define model
    model_dict = {'resnet18': FCNResNet,'resnet34': FCNResNet,'resnet50': FCNResNet,'resnet101': FCNResNet, 'vgg16': FCNVGG, 'densenet121': FCNDenseNet, 'unet': UNet}
    model = model_dict[cfg.backbone](pretrained=cfg.pretrained, num_classes=3, backbone=cfg.backbone, checkpoint=cfg.checkpoint) # modified
    model.load_state_dict(torch.load('/cluster/home/it_stu167/wwj/classification_after_crop/result/CACSeg/resnet18/ACSConv/200911_104150_pretrained/model.dat'))
    
    # convert to counterparts and load pretrained weights according to various convolution
    if cfg.conv=='ACSConv':
        model  = model_to_syncbn(ACSConverter(model))
    if cfg.conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if cfg.conv=='Conv3d':
        if cfg.pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
            if cfg.pretrained_3d == 'video':
                model = load_video_pretrained_weights(model, env.video_resnet18_pretrain_path)
            elif cfg.pretrained_3d == 'mednet':
                model = load_mednet_pretrained_weights(model, env.mednet_resnet18_pretrain_path)
    # print(model)

    # train and test the model
    train(model=model, train_set=None, valid_set=None, test_set=test_set, save=save_path, n_epochs=n_epochs)

    print('Done!')



def train(model, train_set, test_set, save, valid_set, n_epochs):
    '''
    Main training function
    '''
    # Dataloaders

    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    # Model on cuda
    model = to_device(model)
    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:       
        print('multi gpus')
        if cfg.use_syncbn:
            print('Using sync-bn')
            model_wrapper = DataParallelWithCallback(model).cuda()
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()





    # train and test the model
    best_dice_global = 0
    global iteration
    iteration = 0
    for epoch in range(1):


        # test epoch
        test_meters = test_epoch(
            model=model_wrapper,
            loader=test_loader,
            epoch=epoch,
            is_test=True,  # valid
            writer = None
        )




def test_epoch(model, loader, epoch, print_freq=1, is_test=True, writer=None):
    '''
    One test epoch
    '''
    savpath='/cluster/home/it_stu167/wwj/classification_after_crop/result/visualization_3d/'
    meters = MultiAverageMeter()
    model.eval()
    intersection = 0
    union = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (x, y, name) in enumerate(loader):
            x = x.float()
            x = to_var(x)
            
            # forward
            pred_logit = model(x)
            
            # calculate metrics
            pred_classes = pred_logit.argmax(1)
            num = pred_classes.size(0)
            # print(type(y[0,0]))
            # print(name)
            for it in range(num):
                figure = plotly_3d_scan_to_html(scan=y[it,0].numpy(), filename=savpath+name[it]+'_gt.html', title=savpath+name[it]+'_gt.html')
                figure = plotly_3d_scan_to_html(scan=pred_classes[it].cpu().numpy(), filename=savpath+name[it]+'_pd.html', title=savpath+name[it]+'_pd.html')
                
            print(batch_idx)
            

    return None

if __name__ == '__main__':
    fire.Fire(main)

#scan=np.load('/cluster/home/it_stu167/wwj/adrenal/roi/roi/labels/Adr104_L.npz')['arr']

#scan=scan[36:84,36:84,36:84]
#print(scan.shape)
#figure=plotly_3d_scan_to_html(scan=scan, filename='tmp.html')