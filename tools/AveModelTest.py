#Get the global model by all reduce
import torch
from torch import distributed
from torch.optim.optimizer import Optimizer,required

class AVEModelTest(Optimizer):
    def __init__(self,params,model,args=None):
        defaults=dict()
        super(AVEModelTest,self).__init__(params,defaults)
        self.model=model
        
        self.rank=args.rank
        self.node_num=args.world_size

        if self.rank>self.node_num:
            raise ValueError("Rank more than world size")

    def __setstate__(self,state):
        super(AVEModelTest,self).__setstate__(state)

    def ave_params(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state=self.state[p]

                for i in range(self.node_num):
                    if 'param_buff'+str(i) not in param_state:
                        param_state['param_buff'+str(i)]=torch.zeros_like(p.data)

                for i in range(self.node_num):
                    param_state['param_buff'+str(i)]=torch.zeros_like(p.data).cuda()
                param_state['param_buff'+str(self.rank)]=torch.clone(p.data)
                for i in range(self.node_num):
                    distributed.all_reduce(param_state['param_buff'+str(i)])
        distributed.barrier()

        for group in self.param_groups:
            for p in group['params']:
                param_state=self.state[p]

                tem_state=torch.zeros_like(p.data).cuda()
                for i in range(self.node_num):
                    tem_state.add_(param_state['param_buff'+str(i)].div(torch.Tensor([self.node_num]).cuda()))

                p.data=torch.clone(tem_state)

    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure
        self.ave_params()
        return loss