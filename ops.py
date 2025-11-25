import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=None, c_0=None, num=0.5, kmult=0.007, name=None, device=None,
                 amplifier=.1, theta=np.inf, damping=.4, eps=1e-5, weight_decay=0, kappa=0.9, stratified=True):

        defaults = dict(idx=idx, lr=lr, c_0=c_0, w=w, num=num, kmult=kmult, agents=agents, name=name,
                        device=device,
                        amplifier=amplifier, theta=theta, damping=damping, eps=eps, weight_decay=weight_decay,
                        kappa=kappa, lamb=lr, stratified=stratified)

        super(Base, self).__init__(params, defaults)

    def collect_params(self, lr=False):
        for group in self.param_groups:
            grads = []
            vars = []
            if lr:
                return group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                vars.append(p.data.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return vars, grads

    def step(self):
        pass


class Dingtie(Base):
    def __init__(self, *args, **kwargs):
        super(Dingtie, self).__init__(*args, **kwargs)
        self.agent_zeta = {}

    def collect_s(self):
        for group in self.param_groups:
            return group["s"]

    def collect_prev_grad(self):
        for group in self.param_groups:
            return group["prev_grad"]


    def step(self, k=None, vars=None, grads=None, s_all=None, prev_grad_all=None, ns=None, budget=None, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            s_list = []
            prev_grad_list = []

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                if k == 0:
                    s_t = s_all[idx][i + sub].to(device)
                    summat_x = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * (vars[j][i + sub].to(device) - p.data)

                    p.data = p.data + summat_x - lr * p.grad.data

                else:
                    summat_s = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_s += w[idx, j] * s_all[j][i + sub].to(device)

                    s_t = summat_s + p.grad.data - prev_grad_all[idx][i + sub].to(
                            device)

                    summat_x = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * vars[j][i + sub].to(device) 

                    p.data = summat_x - lr * s_t
                    # if k % 100 == 0:
                    #     print('lr=',lr)

                s_list.append(s_t.clone().detach())
                prev_grad_list.append(p.grad.data.clone().detach())

            group["lr"] = lr
            group["s"] = s_list
            group["prev_grad"] = prev_grad_list

        return loss

class DCGT(Base):
    def __init__(self, *args, **kwargs):
        super(DCGT, self).__init__(*args, **kwargs)
        self.agent_zeta = {}
        

    def collect_s(self):
        for group in self.param_groups:
            return group["s"]

    def collect_prev_grad(self):
        for group in self.param_groups:
            return group["prev_grad"]


    def step(self, k=None, vars=None, grads=None, s_all=None, prev_grad_all=None, ns=None, budget=None, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        K0 = 100  
        theta = 1e-4
        # lr = 1e-2

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]
            lr = group['lr']
            c_0 = group['c_0']
            

            s_list = []
            prev_grad_list = []

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                if k == 0:
                    s_t = s_all[idx][i + sub].to(device)
                    summat_x = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * (vars[j][i + sub].to(device) - p.data)
                    
                    p.data = p.data + summat_x - lr * p.grad.data

                else:
                    summat_s = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_s += w[idx, j] * s_all[j][i + sub].to(device)
                                

                    s_t = summat_s + p.grad.data - prev_grad_all[idx][i + sub].to(device)
                           

                    s_t_norm = torch.norm(s_t) 
                    # if s_t_norm > c_0 and s_t_norm <= 1:
                    if s_t_norm > c_0:
                        lr = lr * (c_0 / s_t_norm)
                        # print('c_0 / s_t_norm=',c_0 / s_t_norm)

                    summat_x = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * vars[j][i + sub].to(device)

                        # if k % K0 == 0:
                        #     noise = theta * torch.randn_like(summat_x)  # 高斯噪声
                        #     summat_x += noise

                    p.data = summat_x - lr * s_t
                    # if k % 100 == 0:
                    #     print('lr=',lr)

                s_list.append(s_t.clone().detach())
                prev_grad_list.append(p.grad.data.clone().detach())

            # group["lr"] = lr
            group["s"] = s_list
            group["prev_grad"] = prev_grad_list

        return loss


class CDSGD(Base):
    def __init__(self, *args, **kwargs):
        super(CDSGD, self).__init__(*args, **kwargs)

    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            num = group['num']
            kmult = group['kmult']
            w = group['w']
            agents = group["agents"]
            device=group["device"]

            lr = num / (kmult * k + 1)
            group["lr"] = lr
            
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))
                p.data = summat - lr * p.grad.data
        return loss
