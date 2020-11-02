# encoding: utf-8

import _init_paths
import csv
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
from cac_dataset_tta2 import CACTwoClassDataset
from mylib.utils import MultiAverageMeter, save_model, log_results, to_var, set_seed, \
        to_device, initialize, categorical_to_one_hot, copy_file_backup, redirect_stdout, \
        model_to_syncbn

from cac_config import CACClassConfig as cfg
from cac_config import CACEnv as env

from resnet1 import ClsResNet
from densenet import ClsDenseNet
from vgg import ClsVGG
from acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter
from load_pretrained_weights_funcs import load_mednet_pretrained_weights, load_video_pretrained_weights
from sklearn.metrics import confusion_matrix

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
    test_set = CACTwoClassDataset(crop_size=[48,48,48], data_path=env.data, datatype=3, fill_with=-1)

    # Define model
    model_dict = {'resnet18': ClsResNet, 'resnet34': ClsResNet, 'resnet50': ClsResNet, 'resnet101': ClsResNet, 'resnet152': ClsResNet, 'vgg16': ClsVGG, 'densenet121': ClsDenseNet}
    model = model_dict[cfg.backbone](pretrained=cfg.pretrained, num_classes=2, backbone=cfg.backbone, checkpoint=cfg.checkpoint, pooling=cfg.pooling)

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
    # torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))
    # torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    model_path='/cluster/home/it_stu167/wwj/classification_after_crop/ACSConv/experiments/lidc/model.dat'
    model.load_state_dict(torch.load(model_path))
    # train and test the model
    train(model=model,  test_set=test_set, save=save_path, n_epochs=n_epochs)

    print('Done!')



def train(model,  test_set, save, n_epochs):
    '''
    Main training function
    '''
    # Dataloaders
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    # Model on cuda
    model = to_device(model)

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    print('num_of_cuda:',torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:     
        print('multi-gpus')  
        if cfg.use_syncbn:
            print('Using sync-bn')
            model_wrapper = DataParallelWithCallback(model).cuda()
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones,gamma=cfg.gamma) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_lr, eps=cfg.eps)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr, epochs=n_epochs, steps_per_epoch=len(train_loader), 
    #                                                div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
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
        
        print('learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
    
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

        
        # if cfg.save_all:

    # end 
    writer.close()



def train_epoch(model, loader, optimizer,scheduler, epoch, n_epochs, print_freq=1, writer=None):
    '''
    One training epoch
    '''
    meters = MultiAverageMeter()
    pred_all_probs=[]
    gt_classes=[]
    # Model on train mode
    model.train()
    global iteration
    end = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # Create vaiables
        x = x.float()    # turn halftensor to floattensor
        x = to_var(x)
        y = to_var(y)
        # forward and backward
        pred_logits = model(x)
        loss = F.cross_entropy(pred_logits, y)     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # calculate metrics
        pred_class = pred_logits.max(-1)[1]
        pred_probs = pred_logits.softmax(-1)
        pred_all_probs.append(pred_probs.cpu())
        gt_classes.append(y.cpu())
        batch_size = y.size(0)
        num_classes = pred_logits.size(1)
        same = pred_class==y
        acc = same.sum().item() / batch_size
        accs = torch.zeros(num_classes)
        for num_class in range(num_classes):  # calculate recall for each class
            accs[num_class] = (same * (y==num_class)).sum().item() / ((y==num_class).sum().item()+1e-6)

        # log
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        with open(os.path.join(cfg.save, 'loss_logs.csv'), 'a') as f:
            f.write('%09d,%0.6f,\n'%((iteration + 1),loss.item(),))
        iteration += 1

        logs = [loss.item(), acc]+ \
                            [accs[i].item() for i in range(len(accs))]+ [lr]+\
                            [time.time() - end]
        meters.update(logs, batch_size)   # calculate various index above
        end = time.time()
        # print stats
        print_freq = 2 // meters.val[-1] + 1
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                'ACC %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
            ])
            print(res)
    gt_classes = torch.cat(gt_classes, 0).numpy()
    pred_all_probs = torch.cat(pred_all_probs, 0).detach().numpy()
    auc = roc_auc_score(gt_classes, pred_all_probs[:,1])
    print('auc:', auc)
    return meters.avg[:-1]+[auc]


def test_epoch(model, loader, epoch, print_freq=1, is_test=True, writer=None):
    '''
    One test epoch
    '''
    meters = MultiAverageMeter()
    # Model on eval mode
    model.eval()
    gt_classes = []
    pred_all_probs = []
    pred_all_classes = []
    end = time.time()
    names=[]
    with torch.no_grad():
        for batch_idx, (x, y, name) in enumerate(loader):
            for n in name:
                names.append(n)
            x = x.float()   # b*5*3*48*48*48
            x = to_var(x)   
            y = to_var(y)   # b*5
            batch_size = y.size(0)
            # forward
            rep=x.shape[1]
            loss=0
            if cfg.tta_type=='avg':
                pred_probs=torch.zeros((batch_size,2))
                pred_probs = pred_probs.float()
                pred_probs = to_var(pred_probs)
                for i in range(rep):
                    pred_logits = model(x[:,i]) # b*2
                    loss += F.cross_entropy(pred_logits, y).item()
                    pred_probs = pred_probs + pred_logits.softmax(-1)
                loss /= float(rep)
                pred_probs = pred_probs/float(rep)
            elif cfg.tta_type=='max':
                pred_probs=torch.zeros((batch_size,rep))
                pred_probs = pred_probs.float()
                pred_probs = to_var(pred_probs)
                for i in range(rep):
                    pred_logits = model(x[:,i]) # b*2
                    loss += F.cross_entropy(pred_logits, y).item()
                    pred_probs[:,i] = pred_logits.softmax(-1)[:,1]
                loss /= float(rep)
                pred_probs=torch.max(pred_probs,1)[0]
                pred_probs=torch.unsqueeze(pred_probs,1) # b*1
                pred_probs=torch.cat((to_var(torch.ones((batch_size,1)))-pred_probs,pred_probs),1) # b*2
            # calculate metrics
            pred_class = pred_probs.max(-1)[1]
            pred_all_classes.append(pred_class.cpu())
            
            pred_all_probs.append(pred_probs.cpu())
            gt_classes.append(y.cpu())

            num_classes = pred_probs.size(1)
            same = pred_class==y
            acc = same.sum().item() / batch_size
            accs = torch.zeros(num_classes)
            for num_class in range(num_classes):
                accs[num_class] = (same * (y==num_class)).sum().item() / ((y==num_class).sum().item()+ 1e-6)

            logs = [loss, acc]+ \
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
    gt_classes = torch.cat(gt_classes, 0).numpy()
    pred_all_classes = torch.cat(pred_all_classes, 0).numpy()
    np.save('/cluster/home/it_stu167/wwj/classification_after_crop/ACSConv/experiments/lidc/gt.npy',gt_classes)
    np.save('/cluster/home/it_stu167/wwj/classification_after_crop/ACSConv/experiments/lidc/pred.npy',pred_all_classes)
    pred_all_probs = torch.cat(pred_all_probs, 0).numpy()
    print('confusion matrix:',confusion_matrix(gt_classes, pred_all_classes))
    print('names:',names)

    f = open('/cluster/home/it_stu167/wwj/classification_after_crop/ACSConv/experiments/lidc/result_prob_full.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["id","prediction","probability","groundtruth"])

    for k in range(len(names)):
        csv_writer.writerow([names[k],pred_all_classes[k],pred_all_probs[k][1],gt_classes[k]])
    f.close()

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

# def log_probs(save, epoch, log_dict):
#     with open(os.path.join(save, 'logs.csv'), 'a') as f:
#         f.write('%03d,'%((epoch + 1),))
#         for value in log_dict.values():
#             f.write('%0.6f,' % (value,))
#         f.write('\n')


if __name__ == '__main__':
    fire.Fire(main)
