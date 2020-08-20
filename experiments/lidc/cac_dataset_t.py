from torch.utils.data import Dataset
import random
import os
import numpy as np
import pandas as pd

from mylib.voxel_transform import rotation, reflection, crop, crop_at_zyx_with_dhw, random_center
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
        self.fill_with=fill_with
        # info = pd.read_csv(os.path.join(data_path, 'info/lidc_nodule_info_new_with_subset.csv'), index_col='index')
        # self.info = info[info['malignancy_mode']!=3]
        if datatype==0:
            with open(os.path.join(data_path,'fulldata_train1.txt'),'r') as f:
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
        self.transform = ClassTransform(crop_size, data_path, move=2, datatype=datatype)
        # self.map = {'1':0, '2':0, '4':1, '5':1}
    def __getitem__(self, index):
        name=self.names[index]
        
        ct=np.load(os.path.join(self.data_path, 'roi/roi/img_after_windowing', name+'.npy')) # 3*120*120*120
        # if loc=='L':
        #     center=np.load(os.path.join(self.data_path,'center',name+'_left.npy'))
        # elif loc=='R':
        #     center=np.load(os.path.join(self.data_path,'center',name+'_right.npy'))
        return self.transform(ct,name) # 3*48*48*48



    def __len__(self):
        return len(self.names)


class ClassTransform:
    def __init__(self, size, data_path, move=None, datatype=0):
        self.size = size
        self.move = move
        # self.copy_channels = copy_channels
        self.datatype = datatype
        self.data_path=data_path
    def __call__(self, voxel, name):
        shape = voxel.shape[1:]  # 3 dims
        if self.datatype==0:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            y= np.load(os.path.join(self.data_path, 'roi/roi/labels', name+'.npz'))['arr']
            y= y[center[0] - self.size[0] // 2:center[0] + self.size[0] // 2,
              center[1] - self.size[1] // 2:center[1] + self.size[1] // 2,
              center[2] - self.size[2] // 2:center[2] + self.size[2] // 2]
            loc=name.split('_')[-1]
            if loc=='L':
                lb = 1 if 3 in y else 0
            elif loc=='R':
                lb = 1 if 4 in y else 0
            else:
                print('error')
            angle = np.random.randint(4, size=3)
            voxel_ret = np.stack([rotation(voxel_ret[0], angle=angle),rotation(voxel_ret[1], angle=angle),rotation(voxel_ret[2], angle=angle)],0)

            axis = np.random.randint(4) - 1
            voxel_ret = np.stack([reflection(voxel_ret[0], axis=axis),reflection(voxel_ret[1], axis=axis),reflection(voxel_ret[2], axis=axis)],0)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            y= np.load(os.path.join(self.data_path, 'roi/roi/labels', name+'.npz'))['arr']
            y= y[center[0] - self.size[0] // 2:center[0] + self.size[0] // 2,
              center[1] - self.size[1] // 2:center[1] + self.size[1] // 2,
              center[2] - self.size[2] // 2:center[2] + self.size[2] // 2]
            loc=name.split('_')[-1]
            if loc=='L':
                lb = 1 if 3 in y else 0
            elif loc=='R':
                lb = 1 if 4 in y else 0
            else:
                print('error')
        return voxel_ret,lb
        # if self.copy_channels:
        #     return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32)
        # else:
        #     return np.expand_dims(voxel_ret, 0).astype(np.float32)


