from torch.utils.data import Dataset
import random
import os
import numpy as np
import pandas as pd

from mylib.voxel_transform import rotation, reflection, crop, crop_at_zyx_with_dhw, random_center
from mylib.utils import _triple, categorical_to_one_hot




class LIDCSegDataset(Dataset):
    def __init__(self, crop_size, move, data_path, train=True, copy_channels=True):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        self.move = move
        
        if train:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['train'].\
                dropna().map(lambda x: os.path.join(self.data_path, 'nodule', x)).values
        else:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['test'].\
                dropna().map(lambda x: os.path.join(self.data_path, 'nodule', x)).values
        self.transform = Transform(crop_size, move, train, copy_channels)

    def __getitem__(self, index):
        with np.load(self.names[index]) as npz:
            return self.transform(npz['voxel'], npz['answer1'])


    def __len__(self):
        return len(self.names)

class Transform:
    def __init__(self, size, move=None, train=True, copy_channels=True):
        self.size = _triple(size)
        self.move = move
        self.copy_channels = copy_channels
        self.train = train

    def __call__(self, voxel, seg):
        shape = voxel.shape
        voxel = voxel/255. - 1
        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)
            
            angle = np.random.randint(4, size=3)
            voxel_ret = rotation(voxel_ret, angle=angle)
            seg_ret = rotation(seg_ret, angle=angle)

            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis)
            seg_ret = reflection(seg_ret, axis=axis)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)
            
        if self.copy_channels:
            return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32), \
                    np.expand_dims(seg_ret,0).astype(np.float32)
        else:
            return np.expand_dims(voxel_ret, 0).astype(np.float32), \
                    np.expand_dims(seg_ret,0).astype(np.float32)


class CACTwoClassDataset(Dataset):
    def __init__(self, crop_size, data_path, train=True,fill_with=-1):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        # self.move = move
        self.names=[]
        self.fill_with=fill_with
        # info = pd.read_csv(os.path.join(data_path, 'info/lidc_nodule_info_new_with_subset.csv'), index_col='index')
        # self.info = info[info['malignancy_mode']!=3]
        if train:
            with open(os.path.join(data_path,'cacdata_train.txt'),'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    self.names.append(line)
        else:
            with open(os.path.join(data_path,'cacdata_test.txt'),'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    self.names.append(line)
        # self.transform = ClassTransform(crop_size, move, train, copy_channels)
        # self.map = {'1':0, '2':0, '4':1, '5':1}
    def __getitem__(self, index):
        name,loc,label=self.names[index].split(' ')
        ct=np.load(os.path.join(self.data_path, 'x', name+'.npy')) # 3*512*512*d
        if loc=='l':
            center=np.load(os.path.join(self.data_path,'center',name+'_left.npy'))
        elif loc=='r':
            center=np.load(os.path.join(self.data_path,'center',name+'_right.npy'))
        return crop_at_zyx_with_dhw(ct,np.squeeze(center),self.crop_size,self.fill_with),int(label)


    def __len__(self):
        return len(self.names)


class ClassTransform:
    def __init__(self, size, move=None, train=True, copy_channels=True):
        self.size = _triple(size)
        self.move = move
        self.copy_channels = copy_channels
        self.train = train

    def __call__(self, voxel):
        shape = voxel.shape
        voxel = voxel/255. - 1
        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            
            angle = np.random.randint(4, size=3)
            voxel_ret = rotation(voxel_ret, angle=angle)

            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
        if self.copy_channels:
            return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32)
        else:
            return np.expand_dims(voxel_ret, 0).astype(np.float32)


