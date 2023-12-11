import math
import copy
import torch
from torch import pi as pi
from torch import sin, exp
from utils.HopeGrad import hopegrad, mixed_part, cleargrad

def cald(y,x,idx):
    '''calculate ay/a(x_idx)'''
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0][:,idx-1].reshape(-1, 1)

def check_grad(vars:list):
    for var in vars:
        assert var.grad==None, "The gradient should be cleared first."
        assert not hasattr(var, 'hope_grad'), "The gradient should be cleared first."

def create_flattened_coords(shape) -> torch.Tensor:
    parameter = []
    dim = 1
    for i in range(len(shape)):
        minimum,maximum,num = shape[i]
        parameter.append(torch.linspace(minimum,maximum,num))
        dim *= num
    coords = torch.stack(torch.meshgrid(parameter),axis=-1)
    flattened_coords = coords.reshape(dim,len(shape))
    return flattened_coords

def normalize_data(data, scale_min, scale_max, data_min:float=None, data_max:float=None):
    if data_min==None or data_max==None:
        data_min, data_max = data.min(), data.max()
    if scale_min=='none' or scale_max=='none':
        side_info = {'scale_min':None, 'scale_max':None, 'data_min':float(data_min), 'data_max':float(data_max)}
    else:
        data = (data - data_min)/(data_max - data_min)
        data = data*(scale_max - scale_min) + scale_min
        side_info = {'scale_min':float(scale_min), 'scale_max':float(scale_max), 'data_min':float(data_min), 'data_max':float(data_max)}
    return data, side_info

def invnormalize_data(data, scale_min, scale_max, data_min, data_max):
    if scale_min!=None and scale_max!=None:
        data = (data - scale_min)/(scale_max - scale_min)
        data = data*(data_max - data_min) + data_min
    return data

class BaseSampler:
    def __init__(self, batch_size: int, epochs:int, device:str='cpu'):
        self.batch_size = int(batch_size)
        self.epochs = epochs
        self.device = device
        self.evaled_epochs = []

    def judge_eval(self, eval_epoch):
        if self.epochs_count%eval_epoch==0 and self.epochs_count!=0 and not (self.epochs_count in self.evaled_epochs):
            self.evaled_epochs.append(self.epochs_count)
            return True
        elif self.index>=self.pop_size and self.epochs_count>=self.epochs-1:
            self.epochs_count = self.epochs
            return True
        else:
            return False

    def __len__(self):
        return self.epochs*math.ceil(self.pop_size/self.batch_size)

    def __iter__(self):
        self.index = 0
        self.epochs_count = 0
        return self

# Harmonic oscillator system: u=sin(t) [0,2*pi]
# utttt + 2utt + u = 0
# u(0) = 0, ut(0) = 1, utt(0) = 0
class HarmSampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.shape = [[0,2*pi,128]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.boundary_init = create_flattened_coords([[0,0,1]]).to(device)
        self.fn_label = sin(self.coords)              # batch label
        self.fn_label, self.side_info = normalize_data(self.fn_label, scale_min='none', scale_max='none')
        self.pop_size = self.coords.shape[0]

    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_fn_label = self.fn_label[sampled_idxs, :]
            self.index += self.batch_size
            sampled_init = copy.deepcopy(self.boundary_init).to(self.device)
            return sampled_coords, sampled_fn_label, sampled_init
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration

def HarmLoss(net, sampled_data:torch.tensor):
    t, u_hat, t_init = sampled_data
    t.requires_grad, t_init.requires_grad = True, True
    check_grad([t, t_init])

    u = net(t)
    hopegrad(u, order=4, mixed=0)                           # calculate all the derivatives [ut,utt,uttt,utttt]
    utt, utttt = t.hope_grad[2], t.hope_grad[4]
    loss_pde = (utttt + 2*utt + u)**2                       # pde error: utttt + 2utt + u = 0   u,ut,utt,uttt,utttt

    u_init = net(t_init)
    hopegrad(u_init, order=2, mixed=0)                      # calculate all the derivatives [ut,utt]
    ut_init, utt_init = t_init.hope_grad[1], t_init.hope_grad[2]
    loss_init = u_init**2 + (ut_init - 1)**2 + utt_init**2  # init error: u(0) = 0, ut(0) = 1, utt(0) = 0

    loss = 5*torch.mean(loss_pde)+torch.mean(loss_init)
    fn_loss = torch.mean((u - u_hat)**2)

    loss.backward()
    cleargrad(u)                                            # clear all the high-order derivatives to save gpu
    
    return loss, fn_loss

def HarmLossPINN(net, sampled_data:torch.tensor):
    t, u_hat, t_init = sampled_data
    t.requires_grad, t_init.requires_grad = True, True
    check_grad([t, t_init])

    u = net(t)
    ut = cald(u, t, 1)
    utt = cald(ut, t, 1)
    uttt = cald(utt, t, 1)
    utttt = cald(uttt, t, 1)
    loss_pde = (utttt + 2*utt + u)**2                       # pde error: utttt + 2utt + u = 0   u,ut,utt,uttt,utttt

    u_init = net(t_init)
    ut_init = cald(u_init, t_init, 1)
    utt_init = cald(ut_init, t_init, 1)
    loss_init = u_init**2 + (ut_init - 1)**2 + utt_init**2  # init error: u(0) = 0, ut(0) = 1, utt(0) = 0

    loss = 5*torch.mean(loss_pde)+torch.mean(loss_init)
    fn_loss = torch.mean((u - u_hat)**2)

    loss.backward()
    
    return loss, fn_loss

# Helmholtz Equation：u=e**(-x-y), x,y in [0,1]
# uxxxxxxxx + 4uxxxxxxyy + 6uxxxxyyyy + 4uxxyyyyyy + uyyyyyyyy + u = 17*e**(-x-y)
# u(0,y) = e**(-y), u(1,y) = e**(-1-y)
# u(x,0) = e**(-x), u(x,1) = e**(-x-1)
class HelmSampler(BaseSampler):
    def __init__(self, batch_size: int, epochs:int, device:str='cpu', normal_min:float=-1, normal_max:float=1) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.shape = [[0,1,128], [0,1,128]]
        self.coords = create_flattened_coords(self.shape).to(device)
        self.boundary_left = create_flattened_coords([[0,0,1], [0,1,128]]).to(device)
        self.boundary_right = create_flattened_coords([[1,1,1], [0,1,128]]).to(device)
        self.boundary_down = create_flattened_coords([[0,1,128], [0,0,1]]).to(device)
        self.boundary_up = create_flattened_coords([[0,1,128], [1,1,1]]).to(device)
        self.fn_label = exp(-self.coords[:,0:1]-self.coords[:,1:2])              # batch label
        self.fn_label, self.side_info = normalize_data(self.fn_label, scale_min='none', scale_max='none')
        self.pop_size = self.coords.shape[0]

    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_fn_label = self.fn_label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_coords, sampled_fn_label, self.boundary_left, self.boundary_right, self.boundary_down, self.boundary_up
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration

def HelmLoss(net, sampled_data:torch.tensor):
    coords, u_hat, bl, br, bd, bu = sampled_data
    coords.requires_grad = True
    check_grad([coords, bl, br, bd, bu])

    u = net(coords)
    hopegrad(u, order=8, mixed=2)           # mixed=2: calculate part of the mixed derivatives

    x, y = coords.data[:,0:1], coords.data[:,1:2]
    uxxxxxxxx = mixed_part(coords.hope_grad['Wt'], coords.hope_grad['vz'], idxs=[1,1,1,1,1,1,1,1])
    uxxxxxxyy = mixed_part(coords.hope_grad['Wt'], coords.hope_grad['vz'], idxs=[1,1,1,1,1,1,2,2])
    uxxxxyyyy = mixed_part(coords.hope_grad['Wt'], coords.hope_grad['vz'], idxs=[1,1,1,1,2,2,2,2])
    uxxyyyyyy = mixed_part(coords.hope_grad['Wt'], coords.hope_grad['vz'], idxs=[1,1,2,2,2,2,2,2])
    uyyyyyyyy = mixed_part(coords.hope_grad['Wt'], coords.hope_grad['vz'], idxs=[2,2,2,2,2,2,2,2])

    # pde error: uxxxxxxxx + 4uxxxxxxyy + 6uxxxxyyyy + 4uxxyyyyyy + uyyyyyyyy + u = 17*e**(-x-y)
    loss_pde = (uxxxxxxxx + 4*uxxxxxxyy + 6*uxxxxyyyy + 4*uxxyyyyyy + uyyyyyyyy + u - 17*exp(-x-y))**2  
    
    lossl = (net(bl)-exp(-bl[:,1:2]))**2    # u(0,y) = e**(-y)
    lossr = (net(br)-exp(-1-br[:,1:2]))**2  # u(1,y) = e**(-1-y)
    lossd = (net(bd)-exp(-bd[:,0:1]))**2    # u(x,0) = e**(-x)
    lossu = (net(bu)-exp(-1-bu[:,0:1]))**2  # u(x,1) = e**(-x-1)

    loss = torch.mean(loss_pde)+torch.mean(lossl+lossr+lossd+lossu)
    fn_loss = torch.mean((u - u_hat)**2)

    loss.backward()
    cleargrad(u)                            # clear all the high-order derivatives to save gpu
    
    return loss, fn_loss

def HelmLossPINN(net, sampled_data:torch.tensor):
    coords, u_hat, bl, br, bd, bu = sampled_data
    coords.requires_grad = True
    check_grad([coords, bl, br, bd, bu])

    u = net(coords)
    x, y = coords.data[:,0:1], coords.data[:,1:2]

    ux = cald(u, coords, 1)
    uxx = cald(ux, coords, 1)
    uxxx = cald(uxx, coords, 1)
    uxxxx = cald(uxxx, coords, 1)
    uxxxxx = cald(uxxxx, coords, 1)
    uxxxxxx = cald(uxxxxx, coords, 1)
    uxxxxxxx = cald(uxxxxxx, coords, 1)
    uxxxxxxxx = cald(uxxxxxxx, coords, 1)

    uxxxxxxy = cald(uxxxxxx, coords, 2)
    uxxxxxxyy = cald(uxxxxxxy, coords, 2)

    uxxxxy = cald(uxxxx, coords, 2)
    uxxxxyy = cald(uxxxxy, coords, 2)
    uxxxxyyy = cald(uxxxxyy, coords, 2)
    uxxxxyyyy = cald(uxxxxyyy, coords, 2)

    uxxy = cald(uxx, coords, 2)
    uxxyy = cald(uxxy, coords, 2)
    uxxyyy = cald(uxxyy, coords, 2)
    uxxyyyy = cald(uxxyyy, coords, 2)
    uxxyyyyy = cald(uxxyyyy, coords, 2)
    uxxyyyyyy = cald(uxxyyyyy, coords, 2)

    uy = cald(u, coords, 2)
    uyy = cald(uy, coords, 2)
    uyyy = cald(uyy, coords, 2)
    uyyyy = cald(uyyy, coords, 2)
    uyyyyy = cald(uyyyy, coords, 2)
    uyyyyyy = cald(uyyyyy, coords, 2)
    uyyyyyyy = cald(uyyyyyy, coords, 2)
    uyyyyyyyy = cald(uyyyyyyy, coords, 2)

    # pde error: uxxxxxxxx + 4uxxxxxxyy + 6uxxxxyyyy + 4uxxyyyyyy + uyyyyyyyy + u = 17*e**(-x-y)
    loss_pde = (uxxxxxxxx + 4*uxxxxxxyy + 6*uxxxxyyyy + 4*uxxyyyyyy + uyyyyyyyy + u - 17*exp(-x-y))**2  
    
    lossl = (net(bl)-exp(-bl[:,1:2]))**2    # u(0,y) = e**(-y)
    lossr = (net(br)-exp(-1-br[:,1:2]))**2  # u(1,y) = e**(-1-y)
    lossd = (net(bd)-exp(-bd[:,0:1]))**2    # u(x,0) = e**(-x)
    lossu = (net(bu)-exp(-1-bu[:,0:1]))**2  # u(x,1) = e**(-x-1)

    loss = torch.mean(loss_pde)+torch.mean(lossl+lossr+lossd+lossu)
    fn_loss = torch.mean((u - u_hat)**2)

    loss.backward()
    
    return loss, fn_loss

SamplerDict = {'Harm': HarmSampler, 'HarmPINN':HarmSampler, 'Helm': HelmSampler, 'HelmPINN': HelmSampler}
LossDict = {'Harm': HarmLoss, 'HarmPINN':HarmLossPINN, 'Helm': HelmLoss, 'HelmPINN':HelmLossPINN}

