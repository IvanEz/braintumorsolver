#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import copy
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# ## Auxiliary functions
def get_diff_operator(dx):
    """[summary]

    Args:
        dx ([type]): [description]

    Returns:
        [type]: [description]
    """
    # filter for #dx
    filter_1 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0],[-0.5, 0, 0.5], [0, 0, 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dy
    filter_2 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, -0.5, 0],[0, 0, 0], [0, 0.5, 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dz
    filter_3 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, -0.5, 0], [0, 0, 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0],[0, 0.5, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dxx
    filter_4 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0],[1., -2, 1.], [0, 0, 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dyy
    filter_5 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 1., 0],[0, -2., 0], [0, 1., 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dzz
    filter_6 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 1., 0], [0, 0, 0]],
    [[0, 0, 0],[0, -2., 0], [0, 0, 0]],
    [[0, 0, 0],[0, 1., 0], [0, 0, 0]]
    ]), 0), 0)
    
    # all filters combined
    Filter_var = np.concatenate([filter_1, filter_2, filter_3, filter_4, filter_5, filter_6], axis=0)
    
    if(torch.cuda.is_available()!=False):
        Filter_var = Variable(torch.FloatTensor(Filter_var).cuda(), requires_grad=False)
    else:
        Filter_var = Variable(torch.FloatTensor(Filter_var), requires_grad=False)
    
    # filter for #dxx -d_1
    filter_7 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0],[1., 0, 1.], [0, 0, 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dyy -d_1
    filter_8 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 1., 0],[0, 0, 0], [0, 1., 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
    ]), 0), 0)
    
    # filter for #dzz -d_1
    filter_9 = np.expand_dims(np.expand_dims(np.array([ 
    [[0, 0, 0],[0, 1., 0], [0, 0, 0]],
    [[0, 0, 0],[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0],[0, 1., 0], [0, 0, 0]]
    ]), 0), 0)

    # all filters combined #w/o d_1
    Filter_var_d = np.concatenate([filter_1, filter_2, filter_3, filter_7, filter_8, filter_9], axis=0)
    
    if(torch.cuda.is_available()):
        Filter_var_d = Variable(torch.FloatTensor(Filter_var_d).cuda(), requires_grad=False)
    else:
        Filter_var_d = Variable(torch.FloatTensor(Filter_var_d), requires_grad=False)

    delz = dx #19.3 / 128
    dely = dx #19.3 / 128
    delx = dx #22.9 / 128

    if (torch.cuda.is_available()):
        d_ = torch.from_numpy(np.array(
            [0.0 / delx, 0.0 / dely, 0.0 / delz, -2. / (delx ** 2), -2. / (dely ** 2), -2. / (delz ** 2)]
            )).type(torch.FloatTensor).cuda()
        dx_ = torch.from_numpy(np.array(
            [1.0 / delx, 1.0 / dely, 1.0 / delz, 1.0 / (delx ** 2), 1.0 / (dely ** 2), 1.0 / (delz ** 2)]
            )).type(torch.FloatTensor).cuda()
    else:
        d_ = torch.from_numpy(np.array(
            [0.0 / delx, 0.0 / dely, 0.0 / delz, -2. / (delx ** 2), -2. / (dely ** 2), -2. / (delz ** 2)]
            )).type(torch.FloatTensor)
        dx_ = torch.from_numpy(np.array(
            [1.0 / delx, 1.0 / dely, 1.0 / delz, 1.0 / (delx ** 2), 1.0 / (dely ** 2), 1.0 / (delz ** 2)]
            )).type(torch.FloatTensor)
        
    return Filter_var, Filter_var_d, dx_.view(1, -1, 1, 1, 1), d_.view(1, -1, 1, 1, 1)


class semi_implicit_solver():
    def __init__(self, Dw, rho, dx, dt, E, nu, gamma, MaxIter, epsilon):
        self.Dw = Dw
        self.rho = rho
        self.dt = dt
        self.Filter_var, self.Filter_var_d, self.dx, self.d_ = get_diff_operator(dx)
        self.E = torch.tensor(E).cuda().view(1, -1, 1, 1, 1)
        self.nu = torch.tensor(nu).cuda().view(1, -1, 1, 1, 1)
        self.gamma = torch.tensor(gamma).cuda()
        self.MaxIter = MaxIter
        self.epsilon = epsilon
        

    def compute_grad_lap(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var, padding=0)
        
        return torch.mul(x, self.dx)


    def compute_lap_grad_wo_d(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var_d, padding=0)
        
        return torch.mul(x, self.dx)


    def compute_grad(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var[:3,...], padding=0)

        return torch.mul(x, self.dx[:,:3,...])


    def compute_lap(self, x, sum_=True):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var[3:,...], padding=0)
        
        if sum_:
            return torch.sum(torch.mul(x, self.dx[:,3:,...]), 1, keepdim=True)
        else:
            return torch.mul(x, self.dx[:,3:,...])
    
    
    def compute_lap_wo_d(self, x, sum_=True):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var_d[3:,...], padding=0)
        
        if sum_:
            return torch.sum(torch.mul(x, self.dx[:,3:,...]), 1, keepdim=True)
        else:
            return torch.mul(x, self.dx[:,3:,...])


    def compute_div(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        grad_x = self.compute_grad(x[:,0:1,...])
        grad_y = self.compute_grad(x[:,1:2,...])
        grad_z = self.compute_grad(x[:,2:3,...])
        
        return grad_x[:,0:1,...]+grad_y[:,1:2,...]+grad_z[:,2:3,...]


    # ## Dirichlet Boundary Condition
    def apply_boundary(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x.data[:,:,0,:,:] = 0.0
        x.data[:,:,:,0,:] = 0.0
        x.data[:,:,:,:,0] = 0.0
        x.data[:,:,-1,:,:] = 0.0
        x.data[:,:,:,-1,:] = 0.0
        x.data[:,:,:,:,-1] = 0.0
        
        return x
    
    
    def update_lame_coeff(self, m, c, phi_brain):
        """[summary]

        Args:
            m ([type]): [description]
            c ([type]): [description]
            phi_brain ([type]): [description]
        """
        ls = torch.cat((m, c), 1)
        # print("ls:", torch.amin(ls), torch.amax(ls))
        self.lamda = torch.sum(torch.div(torch.mul(self.E*self.nu, ls) , (1.+self.nu)*(1.-2.*self.nu)), 1, keepdim=True) #+ (1.0-phi_brain)/1e-17
        self.mu = torch.sum(torch.div(torch.mul(self.E, ls) , 2.*(1. + self.nu)), 1, keepdim=True) #+ (1.0-phi_brain)/1e-17


    def precompute_c(self, c, m, v):
        """[summary]

        Args:
            c ([type]): [description]
            m ([type]): [description]
            v ([type]): [description]

        Returns:
            [type]: [description]
        """
        # soft domain from white and grey matter probabilities
        pw, pg = m[:,0:1,...], m[:,1:2,...]
        phi_tumor = (pw + pg).clamp(min=0, max=1.0)
        
        # get phase field function multiplied by diffusivity
        grad_phi_tumor = self.compute_grad(phi_tumor)
        
        # diffusivity
        D = torch.mul(pw, self.Dw) + torch.mul(pg, self.Dw*0.1)
        grad_D = self.compute_grad(D)
        
        # diffusion term
        grad_lap_c = (1.0-self.epsilon)*self.compute_grad_lap(c)
        diff = torch.mul(D, torch.sum(grad_lap_c[:,3:,...], 1, keepdim=True))+\
            torch.sum(torch.mul(grad_D, grad_lap_c[:,:3,...]), 1, keepdim=True)
        
        # advection term
        adv = -torch.mul(c, self.compute_div(v))-\
            torch.sum(torch.mul(v, grad_lap_c[:,:3,...]), 1, keepdim=True)
        
        # reaction term
        reac = torch.mul(self.rho, c - c**2)
        
        # boundary condition term
        bc = torch.sum(torch.mul(grad_phi_tumor, D*grad_lap_c[:,:3,...]), 1, keepdim=True)
        
        # total term irrespective of c_new
        const_c = torch.mul(c+self.dt*(diff+adv+reac), phi_tumor) + self.dt*bc
        
        return const_c, D, grad_D, phi_tumor, grad_phi_tumor
    
    
    def precompute_m(self, m, v, phi_brain):
        """[summary]

        Args:
            m ([type]): [description]
            v ([type]): [description]
            phi_brain ([type]): [description]

        Returns:
            [type]: [description]
        """
        # get phase field function multiplied by diffusivity
        grad_phi_brain = self.compute_grad(phi_brain)
        
        grad_phi_brain = torch.norm(grad_phi_brain, dim=1, keepdim=True)
        
        # divergence of kroneker product
        div = -torch.cat((self.compute_div(m[:,0:1,...]*v), self.compute_div(m[:,1:2,...]*v), self.compute_div(m[:,2:3,...]*v)), 1)
        
        # boundary condition term
        bc = -torch.mul(m, grad_phi_brain)
        
        # total term irrespective of m_new
        const_m = m + (1.0 - self.epsilon)*self.dt*(div)
        
        return const_m, grad_phi_brain


    def update_c(self, c, v, const_c, D, grad_D, phi_tumor, grad_phi_tumor):
        """[summary]

        Args:
            c ([type]): [description]
            v ([type]): [description]
            const_c ([type]): [description]
            D ([type]): [description]
            grad_D ([type]): [description]
            phi_tumor ([type]): [description]
            grad_phi_tumor ([type]): [description]

        Returns:
            [type]: [description]
        """
        # derivatives without central element in diff operator
        remain_lap_grad = self.epsilon*self.compute_lap_grad_wo_d(c)
        
        # diffusion term
        diff = torch.mul(D, torch.sum(remain_lap_grad[:,3:,...], 1, keepdim=True))+\
            torch.sum(torch.mul(grad_D, remain_lap_grad[:,:3,...]), 1, keepdim=True)
        
        # advection term
        adv = -torch.sum(torch.mul(v, remain_lap_grad[:,:3,...]), 1, keepdim=True)
        
        # boundary condition term
        bc = torch.sum(torch.mul(grad_phi_tumor, D*remain_lap_grad[:,:3,...]), 1, keepdim=True)
        
        # denominator for update
        den_c = (1.0 - self.dt*self.epsilon*D*self.d_[:,3:,...].sum()).pow_(-1)
        
        c = self.dt*(torch.mul(diff+adv, phi_tumor)+bc) + const_c
        
        c = torch.mul(c, den_c)
        c = self.apply_boundary(c)
        c = c.clamp(min=0.0, max=1.0)
        
        return c


    def update_m(self, m, v, c, const_m, phi_brain, m_init):
        """[summary]

        Args:
            m ([type]): [description]
            v ([type]): [description]
            const_m ([type]): [description]
            phi_brain ([type]): [description]
            grad_phi_brain ([type]): [description]

        Returns:
            [type]: [description]
        """
        # div of kroneker product
        grad_mw = self.compute_grad(m[:,0:1,...])
        grad_mg = self.compute_grad(m[:,1:2,...])
        grad_mc = self.compute_grad(m[:,2:3,...])
        
        div_wo_m = -torch.cat((torch.sum(grad_mw*v, 1, keepdim=True), torch.sum(grad_mg*v, 1, keepdim=True), torch.sum(grad_mc*v, 1, keepdim=True)),1)
        
        # denominator for update
        den_m = (1.0 + self.epsilon*self.dt*(self.compute_div(v))).pow_(-1)
        
        # update m
        m = self.epsilon*self.dt*(div_wo_m) + const_m
        
        m = torch.mul(m, den_m)
        m = m.clamp(min=0, max=2.0) # TODO: check sum of m for feasible tissue probability
        m = F.normalize(self.apply_boundary(m), p=1, dim=1) #, (1.-c))
        m[torch.isnan(m)] = 0.0
        m = torch.mul(m, phi_brain)+torch.mul(1.-phi_brain, m_init)
        # TODO: Neuman boundary condition?
        
        return m


    def update_u(self, u, c, phi_brain, grad_phi_brain):
        """[summary]

        Args:
            u ([type]): [description]
            c ([type]): [description]
            phi_brain ([type]): [description]
            grad_phi_brain ([type]): [description]

        Returns:
            [type]: [description]
        """
        # mask = phi_brain>0.9
        # laplacian + gradient of divergence
        lap_ux = self.compute_lap_wo_d(u[:,0:1,...])
        lap_uy = self.compute_lap_wo_d(u[:,1:2,...])
        lap_uz = self.compute_lap_wo_d(u[:,2:3,...])
        
        lap_u = torch.cat((lap_ux, lap_uy, lap_uz), 1)
        div_lap_ = torch.mul((self.lamda+self.mu), lap_u) +\
            torch.mul(self.mu, self.compute_grad(self.compute_div(u)))
        
        u = -div_lap_ + self.gamma*self.compute_grad(c) + self.mu*u*self.d_[:,3:,...]  
        
        # denominator for update
        den_u = ((self.lamda+self.mu)*self.d_[:,3:,...].sum()+self.mu*self.d_[:,3:,...]).pow_(-1)
        
        u = torch.mul(u, den_u)
        u = self.apply_boundary(u)
        
        # TODO: Is this clamping needed for boundary?
        u = torch.mul(u, phi_brain>0.9)
        u[torch.isnan(u)] = 0.0
        
        return u


    def solver_step(self, c, m, u, v, phi_brain, m_0, iter_=2):
        """[summary]

        Args:
            c ([type]): [description]
            m ([type]): [description]
            u ([type]): [description]
            v ([type]): [description]
            phi_brain ([type]): [description]

        Returns:
            [type]: [description]
        """
        # limit value between [0, 1]
        c = c.clamp(min=0.0, max=1.0)
        c_init = copy.deepcopy(c)
        m_init = copy.deepcopy(m)
        u_init = copy.deepcopy(u)
        # print('M:', torch.amin(m),torch.amax(m))

        # redoing everything to properly time it
        const_m, grad_phi_brain = self.precompute_m(m_init, v, phi_brain)
        const_c, D, grad_D, phi_tumor, grad_phi_tumor = self.precompute_c(c_init, m_init, v)

        for _ in range(self.MaxIter):
            
            self.update_lame_coeff(m, c, phi_brain)

            # compute the update of u
            for _ in range(iter_):
                u = self.update_u(u, c, phi_brain, grad_phi_brain)
            v = (u-u_init)/self.dt
            v = v.clamp(min=-5e-4, max=5e-4)  # TODO: check maximum feasible velocity for clamping
            # print('U:', torch.amin(u),torch.amax(u))
            # print('V:', torch.amin(v),torch.amax(v))
            
            # compute the update of m
            for _ in range(iter_):
                m = self.update_m(m, v, c, const_m, phi_brain, m_0)
            # print('M:', torch.amin(m),torch.amax(m))
            
            # compute the update of c
            for _ in range(iter_):
                c = self.update_c(c, v, const_c, D, grad_D, phi_tumor, grad_phi_tumor)
            # print("C:",torch.amin(c),torch.amax(c))
            
        return c, m, u, v