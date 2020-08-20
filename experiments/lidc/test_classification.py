# encoding: utf-8

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
from sklearn.metrics import roc_auc_score

from mylib.sync_batchnorm import DataParallelWithCallback
from cac_dataset_t import CACTwoClassDataset
from mylib.utils import MultiAverageMeter, save_model, log_results, to_var, set_seed, \
        to_device, initialize, categorical_to_one_hot, copy_file_backup, redirect_stdout, \
        model_to_syncbn

from cac_config import CACClassConfig as cfg
from cac_config import CACEnv as env

from resnet import ClsResNet
from densenet import ClsDenseNet
from vgg import ClsVGG
from acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter
from load_pretrained_weights_funcs import load_mednet_pretrained_weights, load_video_pretrained_weights

def main(save_path=cfg.save,      # configuration file
         n_epochs=cfg.n_epochs, 
         seed=cfg.seed
         ):
    # set seed
    if seed is not None:
        set_seed(cfg.seed)
    cudnn.benchmark = True   # improve efficiency
    # back up your code
    os.makedirs(save_path)
    copy_file_backup(save_path)
    redirect_stdout(save_path)

    # Datasets
    valid_set = None
    test_set = CACTwoClassDataset(crop_size=[48,48,48], data_path=env.data, datatype=2, fill_with=-1)

    # Define model
    model_dict = {'resnet18': ClsResNet, 'vgg16': ClsVGG, 'densenet121': ClsDenseNet}
    model = model_dict[cfg.backbone](pretrained=cfg.pretrained, num_classes=2, backbone=cfg.backbone)

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
    torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))
    model_path='/cluster/home/it_stu167/wwj/classification_after_crop/result/CACClass/resnet18/ACSConv/48_1-2_m0/epoch_107/model.dat'
    model.load_state_dict(torch.load(model_path))
    # train and test the model
    train(model=model, valid_set=valid_set, test_set=test_set, save=save_path, n_epochs=n_epochs)

    print('Done!')



def train(model, test_set, save, valid_set, n_epochs):
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
        print('multi-gpus')  
        if cfg.use_syncbn:
            print('Using sync-bn')
            model_wrapper = DataParallelWithCallback(model).cuda()
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()


    # Start logging
    logs = ['loss', 'acc', 'acc0', 'acc1']

    test_logs = ['test_'+log for log in logs]+['test_auc','test_auc_pat']

    log_dict = OrderedDict.fromkeys(test_logs, 0)
    with open(os.path.join(save, 'logs.csv'), 'w') as f:
        f.write('epoch,')
        for key in log_dict.keys():
            f.write(key+',')
        f.write('\n')
    with open(os.path.join(save, 'loss_logs.csv'), 'w') as f:
        f.write('iter,train_loss,\n')
    writer = SummaryWriter(log_dir=os.path.join(save, 'Tensorboard_Results'))

    # train and test the model
    best_auc = 0
    global iteration
    iteration = 0
    for epoch in range(1):
        os.makedirs(os.path.join(cfg.save, 'epoch_{}'.format(epoch)))
        # test epoch
        test_meters = test_epoch(
            model=model_wrapper,
            loader=test_loader,
            epoch=epoch,
            is_test=True,
            writer = writer
        )

        # Log results
        for i, key in enumerate(test_logs):
            log_dict[key] = test_meters[i]
        log_results(save, epoch, log_dict, writer=writer)
        # save model checkpoint
        if log_dict['test_auc'] > best_auc:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
            best_auc = log_dict['test_auc']
            print('New best auc: %.4f' % log_dict['test_auc'])
        else:
            print('Current best auc: %.4f' % best_auc)
    # end 
    writer.close()
    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write(',,,,best auc,%0.5f\n' % (best_auc))
    print('best auc: ', best_auc)

def test_epoch(model, loader, epoch, print_freq=1, is_test=True, writer=None):
    '''
    One test epoch
    '''
    meters = MultiAverageMeter()
    # Model on eval mode
    model.eval()
    gt_classes = []
    pred_all_probs = []
    end = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.float()   
            x = to_var(x)
            y = to_var(y)
            # forward
            pred_logits = model(x)  
            loss = F.cross_entropy(pred_logits, y)
            # calculate metrics
            pred_class = pred_logits.max(-1)[1] # b
            pred_probs = pred_logits.softmax(-1) # b*2
            pred_all_probs.append(pred_probs.cpu()) # n*b*2
            gt_classes.append(y.cpu()) # n*b
            batch_size = y.size(0)
            num_classes = pred_logits.size(1)
            same = pred_class==y
            acc = same.sum().item() / batch_size
            accs = torch.zeros(num_classes)
            for num_class in range(num_classes):
                accs[num_class] = (same * (y==num_class)).sum().item() / ((y==num_class).sum().item()+ 1e-6)

            logs = [loss.item(), acc]+ \
                                [accs[i].item() for i in range(len(accs))]+ \
                                [time.time() - end]
            meters.update(logs, batch_size)   
            end = time.time()


            print_freq = 2 // meters.val[-1] + 1
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                    'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                    'ACC %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                ])
                print(res)
    gt_classes = torch.cat(gt_classes, 0).numpy() # num
    pred_all_probs = torch.cat(pred_all_probs, 0).numpy() # num*2
    auc = roc_auc_score(gt_classes, pred_all_probs[:,1])
    print('auc:', auc)
    # suppose adrenals are in order
    it=0
    pat_probs=[]
    pat_classes=[]
    while it < len(gt_classes):
        pat_classes.append(max(gt_classes[it],gt_classes[it+1]))
        pat_probs.append(max(pred_all_probs[it,1],pred_all_probs[it+1,1]))
        it+=2
    print(len(pat_probs))
    auc_pat=roc_auc_score(pat_classes, pat_probs)
    return meters.avg[:-1]+[auc,auc_pat]


if __name__ == '__main__':
    fire.Fire(main)
