import numpy as np
import os
from skimage import measure

xpath='/cluster/home/it_stu167/wwj/adrenal/x'
ypath='/cluster/home/it_stu167/wwj/adrenal/y'
save_cen='/cluster/home/it_stu167/wwj/adrenal/center'
save_bbox='/cluster/home/it_stu167/wwj/adrenal/bbox'
lcen=np.zeros((1,3))
lcen=lcen.astype(np.int16)
rcen=np.zeros((1,3))
rcen=rcen.astype(np.int16)
lbbox=np.zeros((1,6))
lbbox=lbbox.astype(np.int16)
rbbox=np.zeros((1,6))
rbbox=rbbox.astype(np.int16)

dirs=os.listdir(xpath)
ln=[]
rn=[]
with open('/cluster/home/it_stu167/wwj/adrenal/cac_ann.txt','w') as f:
    for di in dirs:
        name=di[:-4]
        y=np.load(os.path.join(ypath,name+'-seg.npy'))
        lano=1 if 1 in y[:256,:,:] else 0
        rano=1 if 1 in y[256:,:,:] else 0
        y[y==1]=2
        left=y[:256,:,:]
        right=y[256:,:,:]

        tmp1=measure.regionprops(left)
        for i in range(3):
            lcen[0,i]=int(tmp1[0].centroid[i]) # [b,h,w]
        np.save(os.path.join(save_cen,name+'_left.npy'),lcen)
        for i in range(6):
            lbbox[0,i]=tmp1[0].bbox[i]   # [bmin,hmin,wmin,bmax,hmax,wmax]   [bmin,bmax)
        np.save(os.path.join(save_bbox,name+'_left.npy'),lbbox)
        if lbbox[0,3]==256:
            ln.append(name)
        f.writelines(name+' l '+str(lano)+'\n')

        tmp2=measure.regionprops(right)
        for i in range(3):
            rcen[0,i]=int(tmp2[0].centroid[i]) # [b,h,w]
        np.save(os.path.join(save_cen,name+'_right.npy'),rcen)
        for i in range(6):
            rbbox[0,i]=tmp2[0].bbox[i]   # [bmin,hmin,wmin,bmax,hmax,wmax]   [bmin,bmax)
        np.save(os.path.join(save_bbox,name+'_right.npy'),rbbox)
        if rbbox[0,0]==0:
            rn.append(name)
        f.writelines(name+' r '+str(rano)+'\n')

print(len(ln))
print(ln)
print(len(rn))
print(rn)