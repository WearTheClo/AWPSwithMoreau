import torch
from torch import distributed

import os
import time
import numpy as np

class evaluate():
    def __init__(self,args):
        self.loss_item=[]
        self.c_r=[]
        self.epoch=[]
        self.L2Dis=[]
        self.args=args

    def append(self,loss_item,c_r,epoch,L2Dis):
        self.loss_item.append(loss_item)
        self.c_r.append(c_r)
        self.epoch.append(epoch)
        self.L2Dis.append(L2Dis)

    def all_reduce(self):
        value=torch.Tensor([self.loss_item[-1]/self.args.world_size,self.c_r[-1]/self.args.world_size,self.L2Dis[-1]/self.args.world_size]).cuda()
        distributed.all_reduce(value)
        value=value.cpu().float()
        self.loss_item[-1],self.c_r[-1],self.L2Dis[-1]=value[0],value[1],value[2]

    def read_save(self):
        if self.args.rank!=0:
            return
        path_wor=self.args.evaluation+'wor.txt'
        with open(path_wor,'a') as f:
            f.write('epoch:%d Loss:%10.7f Accuracy:%10.7f%% L2Dis:%10.7f\n'%(self.epoch[-1],self.loss_item[-1],self.c_r[-1],self.L2Dis[-1]))
            f.close()

    def time_save(self):
        if self.args.rank!=0:
            return
        end=time.asctime()
        end_time=(int(end[11:13]),int(end[14:16]),int(end[17:19]))
        time_cost=0
        time_cost+=(end_time[0]-self.args.begin_time[0])*3600
        time_cost+=(end_time[1]-self.args.begin_time[1])*60
        time_cost+=end_time[2]-self.args.begin_time[2]
        if time_cost<0:
            time_cost+=86400
        time_cost=np.array(time_cost)

        print("Time cost:%d"%time_cost)
        np.save(self.args.evaluation+'Time.npy',time_cost)

    def np_save(self):
        if self.args.rank!=0:
            return
        e=np.array(self.epoch)
        l=np.array(self.loss_item)
        c=np.array(self.c_r)
        d=np.array(self.L2Dis)

        np.save(self.args.evaluation+'Epoch.npy',e)
        np.save(self.args.evaluation+'Loss.npy',l)
        np.save(self.args.evaluation+'CR.npy',c)
        np.save(self.args.evaluation+'Distance.npy',d)
