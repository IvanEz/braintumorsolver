#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
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
        
    return Filter_var, Filter_var_d, dx_, d_


class semi_implicit_solver():
    def __init__(self, Dw, rho, dx, dt, lamda, mu, gamma, MaxIter, epsilon):
        self.Dw = Dw
        self.rho = rho
        self.dt = dt
        self.Filter_var, self.Filter_var_d, self.dx, self.d_ = get_diff_operator(dx)
        self.lamda = torch.tensor(lamda).cuda()
        self.mu = torch.tensor(mu).cuda()
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
        
        return torch.mul(x, self.dx.view(1, -1, 1, 1, 1))


    def compute_lap_grad_wo_d(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var_d, padding=0)
        
        return torch.mul(x, self.dx.view(1, -1, 1, 1, 1))


    def compute_grad(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = F.pad(x, (1,1,1,1,1,1), mode='constant')
        x = F.conv3d(x, self.Filter_var[:3,...], padding=0)

        return torch.mul(x, self.dx[:3].view(1, -1, 1, 1, 1))


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
            return torch.mul(torch.sum(x, 1, keepdim=True), self.dx[3:].view(1, -1, 1, 1, 1))
        else:
            return torch.mul(x, self.dx[3:].view(1, -1, 1, 1, 1))
    
    
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
            return torch.mul(torch.sum(x, 1, keepdim=True), self.dx[3:].view(1, -1, 1, 1, 1))
        else:
            return torch.mul(x, self.dx[3:].view(1, -1, 1, 1, 1))


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


    def precompute_c(self, c, m, v):
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
        # soft domain from white and grey matter probabilities
        pw, pg = m[:,0:1,...], m[:,1:2,...]
        phi_tumor = pw + pg
        
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
            c ([type]): [description]
            m ([type]): [description]
            u ([type]): [description]
            v ([type]): [description]
            phi_brain ([type]): [description]

        Returns:
            [type]: [description]
        """
        # get phase field function multiplied by diffusivity
        grad_phi_brain = self.compute_grad(phi_brain)
        
        grad_phi_brain = torch.norm(grad_phi_brain, 1, keepdim=True)
        
        # divergence of kroneker product
        mv = torch.cat(((m*v[:,0:1,...]).unsqueeze(2), (m*v[:,1:2,...]).unsqueeze(2), (m*v[:,2:3,...]).unsqueeze(2)), 2)
        div = torch.cat((self.compute_div(mv[:,0,...]), self.compute_div(mv[:,1,...]), self.compute_div(mv[:,2,...])), 1)
        
        # boundary condition term
        bc = torch.mul(m, grad_phi_brain)
        
        # total term irrespective of m_new
        const_m = torch.mul(m, phi_brain) + (1.0 - self.epsilon)*self.dt*(div+bc)
        
        return const_m, grad_phi_brain


    def update_c(self, c, v, const_c, D, grad_D, phi_tumor, grad_phi_tumor):
        """[summary]

        Args:
            c ([type]): [description]
            m ([type]): [description]
            v ([type]): [description]
            const_c ([type]): [description]
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
        
        # LHS divider
        div_c = (1.0 - self.dt*self.epsilon*D*self.d_[3:].sum()).pow_(-1)
        
        c = self.dt*(torch.mul(diff+adv, phi_tumor)+bc) + const_c
        
        c = torch.mul(c, div_c)
        c = self.apply_boundary(c)
        
        return c


    def update_m(self, m, v, const_m, phi_brain, grad_phi_brain):
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
        grad_vx = self.compute_grad(v[:,0:1,...])
        grad_vy = self.compute_grad(v[:,1:2,...])
        grad_vz = self.compute_grad(v[:,2:3,...])
        
        mv = torch.cat(((m*v[:,0:1,...]).unsqueeze(2), (m*v[:,1:2,...]).unsqueeze(2), (m*v[:,2:3,...]).unsqueeze(2)), 2)
        div = -torch.cat((self.compute_div(mv[:,0,...]), self.compute_div(mv[:,1,...]), self.compute_div(mv[:,2,...])), 1)
        lhs = torch.cat((grad_vx[:,0:1,...],grad_vy[:,1:2,...],grad_vz[:,2:3,...]),1)
        div = div + torch.mul(m, lhs)
        div_m = (1.0 + self.epsilon*self.dt*(lhs + grad_phi_brain)).pow_(-1)
        
        # update m
        m = self.epsilon*self.dt*(div) + const_m
        
        m = torch.mul(m, div_m)
        m = self.apply_boundary(m)
        
        return m


    def update_u(self, u, c, phi_brain, grad_phi_brain):
        """[summary]

        Args:
            u ([type]): [description]
            c ([type]): [description]
            const_u ([type]): [description]
            phi_brain ([type]): [description]
            grad_phi_brain ([type]): [description]

        Returns:
            [type]: [description]
        """
        # laplacian + gradient of divergence
        
        lap_ux = self.compute_lap_wo_d(u[:,0:1,...], sum_=False)
        lap_uy = self.compute_lap_wo_d(u[:,1:2,...], sum_=False)
        lap_uz = self.compute_lap_wo_d(u[:,2:3,...], sum_=False)
        
        lap_u = torch.cat((torch.sum(lap_ux, 1, keepdim=True), torch.sum(lap_uy, 1, keepdim=True), torch.sum(lap_uz, 1, keepdim=True)), 1)
        lap_gradiv = torch.mul((self.lamda+self.mu).view(1, -1, 1, 1, 1), lap_u)+\
            torch.mul(self.lamda.view(1, -1, 1, 1, 1), self.compute_grad(self.compute_div(u)))
        
        lap_u_ = torch.cat((lap_ux[:,0:1,...], lap_uy[:,1:2,...], lap_uy[:,2:3,...]), 1)
        
        u = lap_gradiv - lap_u_
        div_u = (-((self.lamda+self.mu+1)*self.d_[3:]).view(1, -1, 1, 1, 1)+grad_phi_brain).pow_(-1)
        
        u = torch.mul(u, div_u)
        u = self.apply_boundary(u)
        
        return u


    def solver_step(self, c, m, u, v, phi_brain):
        """[summary]

        Args:
            c ([type]): [description]
            m ([type]): [description]
            u ([type]): [description]
            v ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        # limit value between [0, 1]
        c = c.clamp(min=0.0, max=1.0)
        c_init = c
        m_init = m
        u_init = u

        #redoing everything to properly time it
        const_m, grad_phi_brain = self.precompute_m(m, v, phi_brain)

        for iter_ in range(self.MaxIter):

            # # compute the update of u
            # u = self.update_u(u, c, phi_brain, grad_phi_brain)
            # v = (u-u_init)/self.dt
            # print('V:', torch.amin(v),torch.amax(v))
            
            # # compute the update of m
            # m = self.update_m(m, v, const_m, phi_brain, grad_phi_brain)
            # m = torch.multiply(m.clamp(min=0, max=1), phi_brain) # TODO: check sum of m
            # print('M:', torch.amin(m),torch.amax(m))
            
            # compute the update of c
            const_c, D, grad_D, phi_tumor, grad_phi_tumor = self.precompute_c(c_init, m, v)
            c = self.update_c(c, v, const_c, D, grad_D, phi_tumor, grad_phi_tumor)
            # print(torch.amin(c),torch.amax(c))
            c = c.clamp(min=0.0, max=1.0)

        return c, m, u, v