import os
import numpy as np
import random

order=[]
with open('/cluster/home/it_stu167/wwj/adrenal-classification/adrenal-classification/order.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line.strip('\n')
        order.append(line)
data=[]
with open('/cluster/home/it_stu167/wwj/adrenal/cac_ann.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line.strip('\n')
        data.append(line)
random.shuffle(order)
with open('/cluster/home/it_stu167/wwj/adrenal/cac_order.txt','w') as f:
    for i in order:
        f.writelines(i+'\n')
random.shuffle(data)
with open('/cluster/home/it_stu167/wwj/adrenal/cacdata_train.txt','w') as f1:
    with open('/cluster/home/it_stu167/wwj/adrenal/cacdata_test.txt','w') as f2:
        for i in data:
            if i.split(' ')[0] in order[:40]:
                f2.writelines(i+'\n')
            else:
                f1.writelines(i+'\n')
