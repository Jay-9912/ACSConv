from torch.utils.data import Dataset
import random
import os
import numpy as np
import pandas as pd

from mylib.voxel_transform import rotation, reflection, crop, crop_at_zyx_with_dhw, random_center_mask_mood, random_center, random_center_mask
from mylib.utils import _triple, categorical_to_one_hot




class CACSegDataset(Dataset):
    def __init__(self, crop_size, data_path, datatype=0):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        self.names=[]       
        # maybe use some transform to augment data
        if datatype==0:
            with open(os.path.join(data_path,'fulldata_train3.txt'),'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    self.names.append(line)
        elif datatype==1:
            with open(os.path.join(data_path,'fulldata_valid1.txt'),'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    self.names.append(line)
        elif datatype==2:
            with open(os.path.join(data_path,'fulldata_test1.txt'),'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    self.names.append(line)    
        else:
            print('error')

    def __getitem__(self, index):
        name=self.names[index]
        loc=name.split('_')[-1]
        ct=np.load(os.path.join(self.data_path, 'roi/roi/img_after_windowing', name+'.npy')) # 3*120*120*120
        # if loc=='L':
        #     center=np.load(os.path.join(self.data_path,'center',name+'_left.npy'))
        # elif loc=='R':
        #     center=np.load(os.path.join(self.data_path,'center',name+'_right.npy'))
        center=np.array([60,60,60])
        x, crop_pos= crop_at_zyx_with_dhw(ct, center, self.crop_size,-1)
        y= np.load(os.path.join(self.data_path, 'roi/roi/labels', name+'.npz'))['arr']
        y= y[crop_pos[0][0]:crop_pos[0][1], crop_pos[1][0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]

        if loc=='L':
            y[y==2]=0
            y[y==4]=0
            y[y==3]=2
        elif loc=='R':
            y[y==1]=0
            y[y==3]=0
            y[y==2]=1
            y[y==4]=2
        else:
            print('error')
        return x,np.expand_dims(y,0).astype(np.float32)  # y: 1*48*48*48
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
    def __init__(self, crop_size, data_path, datatype=0,fill_with=-1):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        # self.move = move
        self.names=[]
        self.labels=[]
        self.fill_with=fill_with
        self.datatype=datatype
        # info = pd.read_csv(os.path.join(data_path, 'info/lidc_nodule_info_new_with_subset.csv'), index_col='index')
        # self.info = info[info['malignancy_mode']!=3]
        if datatype==0:
            train=pd.read_csv(os.path.join(data_path,'mood_train_shuffle.csv'))
            self.names=train['ROI_id'].to_list()
            self.labels=train['ROI_anomaly'].to_list()
        elif datatype==1:
            test=pd.read_csv(os.path.join(data_path,'mood_test.csv'))
            self.names=test['ROI_id'].to_list()
            self.labels=test['ROI_anomaly'].to_list()
        else:
            print('error')

        self.transform = ClassTransform(crop_size, data_path, fill_with)
        # self.map = {'1':0, '2':0, '4':1, '5':1}
    def __getitem__(self, index):
        name=self.names[index]
        lb=1 if self.labels[index] else 0
        ct=np.load(os.path.join(self.data_path, 'roi/ROI-clean', name+'.npz')) 
        if self.datatype==0:
            return self.transform(ct),lb # 3*48*48*48
        else:
            voxel=[]
            for i in range(5): 
                voxel_t=self.transform(ct)
                voxel.append(voxel_t)
            return np.array(voxel),lb # 5*3*48*48*48




    def __len__(self):
        return len(self.names)


class ClassTransform:
    def __init__(self, size, data_path, fill_with):
        self.size = size
        self.data_path = data_path
        self.fill_with = fill_with
    def __call__(self, ct):
        voxel=ct['image']
        voxel=np.expand_dims(voxel,0).repeat(3,axis=0)
        y=ct['coarse_mask']
        shape = voxel.shape[1:]  # 120*120*120
        center = random_center_mask_mood(y)

        voxel_ret,crop_pos = crop_at_zyx_with_dhw(voxel, center, self.size, self.fill_with)
            
        angle = np.random.randint(4, size=3)
        voxel_ret = np.stack([rotation(voxel_ret[0], angle=angle),rotation(voxel_ret[1], angle=angle),rotation(voxel_ret[2], angle=angle)],0)

        axis = np.random.randint(4) - 1
        voxel_ret = np.stack([reflection(voxel_ret[0], axis=axis),reflection(voxel_ret[1], axis=axis),reflection(voxel_ret[2], axis=axis)],0)


        return voxel_ret
        # if self.copy_channels:
        #     return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32)
        # else:
        #     return np.expand_dims(voxel_ret, 0).astype(np.float32)


