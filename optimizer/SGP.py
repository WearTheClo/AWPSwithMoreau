import random
import torch

from torch import distributed
from torch.optim.optimizer import Optimizer,required

class SGP(Optimizer):
    def __init__(self,params,model,lr=required,momentum=False,mr=0.0,args=None):
        if lr is not required and lr<0.0:
            raise ValueError("Invalid learning rate")
        if mr<0.0:
            raise ValueError("Invalid momentum rate")

        if args.type=='ForceRandom':
            self.FR=True
        else:
            self.FR=False

        defaults=dict(lr=lr,mr=mr)
        super(SGP,self).__init__(params,defaults)
        self.model=model
        self.momentum=momentum
        self.B=args.B
        self.iter_cnt=0
        self.rank=args.rank
        self.node_num=args.world_size
        self.communication_size=0

        self.topo=args.topo
        self.weight=[0.0]*self.node_num
        self.WM=torch.Tensor([[0.0]*self.node_num]*self.node_num).cuda()
        self.aux_var=torch.Tensor([1.0]).cuda()
        self.aux_vec=torch.Tensor([0.0]*self.node_num).cuda()
        self.rece_indx=[0]*self.node_num

        if self.rank>self.node_num:
            raise ValueError("Rank more than world size")

        if self.momentum:
            print("Momentum")

        print("Push-SUM")

    def reset_communication_size(self):
        self.communication_size=0

    def add_communication_size(self,each_send_size):
        tem = 0
        for i in range(self.node_num):
            if i==self.rank:
                continue
            if self.WM[i][self.rank]!=0.0:
                tem = tem + 1
        self.communication_size += each_send_size * tem

    def get_acc_communication_size(self):
        return self.communication_size

    def __setstate__(self,state):
        super(AveWeightPSSGD,self).__setstate__(state)

    def ave_weight(self):
        self.weight=[0.0]*self.node_num

        if self.FR and self.iter_cnt % self.B != 0:
            for i in range(self.node_num):
                if self.topo[i][self.rank]==0:
                    continue
                if random.random()<self.topo[i][self.rank]:
                    self.weight[i]=1.0
        else:
            if self.FR:
                idx = 0
            else:
                idx = self.iter_cnt % self.B
            for i in range(self.node_num):
                if self.topo[idx*self.node_num+i][self.rank]==0:
                    continue
                self.weight[i]=1.0

        out=sum(self.weight)
        for i in range(self.node_num):
            if i==self.rank or self.weight[i]==0.0:
                continue
            self.weight[i]=1.0/out
        self.weight[self.rank]=2.0-sum(self.weight)


    def _update_PSSGD_params(self):
        send_size=0

        for i in range(self.node_num):
            for j in range(self.node_num):
                if j==self.rank:
                    self.WM[i][j]=self.weight[i]
                else:
                    self.WM[i][j]=0.0
        distributed.all_reduce(self.WM)

        for i in range(self.node_num):
            if i==self.rank:
                self.aux_vec[i]=float(self.aux_var)
            else:
                self.aux_vec[i]=0.0
        distributed.all_reduce(self.aux_vec)
        tem=torch.Tensor([0.0]).cuda()
        for i in range(self.node_num):
            tem.add_(self.aux_vec[i].mul(self.WM[self.rank][i]))
        self.aux_var=torch.clone(tem)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state=self.state[p]
                d_p=p.grad.data
                if 'param_buff' not in param_state:
                    param_state['param_buff']=torch.clone(p).detach()
                if 'momentum_buff' not in param_state and self.momentum:
                    param_state['momentum_buff']=torch.zeros_like(p.data)

                if self.momentum:
                    param_state['momentum_buff'].mul_(group['mr'])
                    param_state['momentum_buff'].add_(d_p)
                    param_state['param_buff'].add_(param_state['momentum_buff'],alpha=-group['lr'])
                else:
                    param_state['param_buff'].add_(d_p,alpha=-group['lr'])

                for i in range(self.node_num):
                    param_state['param_buff'+str(i)]=torch.zeros_like(p.data).cuda()
                param_state['param_buff'+str(self.rank)]=torch.clone(param_state['param_buff'])

                for i in range(self.node_num):
                    distributed.all_reduce(param_state['param_buff'+str(i)])
                self.add_communication_size(param_state['param_buff'].element_size()*param_state['param_buff'].nelement())
        distributed.barrier()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state=self.state[p]

                    tem_state=torch.zeros_like(p.data).cuda()
                    for i in range(self.node_num):
                        tem_state.add_(param_state['param_buff'+str(i)].mul(self.WM[self.rank][i]))

                    param_state['param_buff']=torch.clone(tem_state)
                    p.data=param_state['param_buff'].div(self.aux_var)

    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure
        self.ave_weight()
        self._update_PSSGD_params()
        self.iter_cnt+=1
        return loss