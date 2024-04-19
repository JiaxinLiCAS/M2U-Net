# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:26:59 2022

@author: 13572
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import os
import scipy
import torch.nn.functional as fun

def init_weights(net, init_type, gain):
    print('in init_weights')
    def init_func(m):
        classname = m.__class__.__name__
        #print(classname,m,'_______')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer):
    print('in init_net')
    net.to(device)  #gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer :
        #print(2,initializer)
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net

    


########################## coupled_U ############################

def Coupled_U(fusion,hrmsi,args, init_type='kaiming', init_gain=0.02,initializer=True ):
    
    net = coupled_U (fusion,hrmsi,args)

    return init_net(net,args.device, init_type, init_gain,initializer)

class coupled_U(nn.Module):
    def __init__(self,fusion,hrmsi,args): 

        super().__init__()
        
        
        self.band=args.band
        self.fusion=fusion #1 C H W
        self.hrmsi=hrmsi #1 C H W
        self.scale=[
                              (  self.fusion.shape[2],self.fusion.shape[3]  ),
                              (  int(self.fusion.shape[2]/2),int(self.fusion.shape[3]/2)  ),
                              (  int(self.fusion.shape[2]/4), int(self.fusion.shape[3]/4) )
                              ]
        #print(self.scale)
        
        
        self.assist1=Assist(self.fusion.shape[1], self.hrmsi.shape[1],args)
        
        self.assist2=Assist(self.band, self.band,args)
        

        self.assist3=Assist(self.band, self.band,args)
        
        self.assist4=Assist(self.band+2, self.band+2,args)
        
        self.assist5=Assist(self.band+2, self.band+2,args)
        
        self.late=nn.Sequential(
            nn.Conv2d(self.band,self.fusion.shape[1],kernel_size=(1,1),stride=1,padding=(0,0)) ,
            nn.Sigmoid()
                                )
        
   
        self.skip1=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip2=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip3=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip4=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        
        
    def forward(self,fusion,hrmsi):
        
        #down
        x1,z1=self.assist1(fusion,hrmsi)
        x2=nn.AdaptiveAvgPool2d(self.scale[1])(x1)
        z2=nn.AdaptiveAvgPool2d(self.scale[1])(z1)
        #print('x1.shape: {}'.format(x1.shape))
        #print('z1.shape: {}'.format(z1.shape))
        #print('x2.shape: {}'.format(x2.shape))
        #print('z2.shape: {}'.format(z2.shape))
        
        
        x3,z3=self.assist2(x2,z2)
        x4=nn.AdaptiveAvgPool2d(self.scale[2])(x3)
        z4=nn.AdaptiveAvgPool2d(self.scale[2])(z3)
        
        #print('x3.shape: {}'.format(x3.shape))
        #print('z3.shape: {}'.format(z3.shape))
        #print('x4.shape: {}'.format(x4.shape))
        #print('z4.shape: {}'.format(z4.shape))
        
        x5,z5=self.assist3(x4,z4)
        #print('x5.shape: {}'.format(x5.shape))
        #print('z5.shape: {}'.format(z5.shape))
        
        #skip
        s1=self.skip1(x3)
        s3=self.skip3(z3)
        
        #print('s1.shape: {}'.format(s1.shape))
        #print('s3.shape: {}'.format(s3.shape))
        
        s2=self.skip2(x1)
        s4=self.skip4(z1)
        
        #print('s2.shape: {}'.format(s2.shape))
        #print('s4.shape: {}'.format(s4.shape))
        
        #up
        x6=nn.Upsample(self.scale[1], mode='bilinear')(x5)
        z6=nn.Upsample(self.scale[1], mode='bilinear')(z5)
        x7,z7=self.assist4(torch.cat([s1,x6],dim=1), torch.cat([s3,z6],dim=1))
        #print('x6.shape: {}'.format(x6.shape))
        #print('z6.shape: {}'.format(z6.shape))
        #print('x7.shape: {}'.format(x7.shape))
        #print('z7.shape: {}'.format(z7.shape))
        
        
        x8=nn.Upsample(self.scale[0], mode='bilinear')(x7)
        z8=nn.Upsample(self.scale[0], mode='bilinear')(z7)
        x9,z9=self.assist5(torch.cat([s2,x8],dim=1), torch.cat([s4,z8],dim=1))
        #print('x8.shape: {}'.format(x8.shape))
        #print('z8.shape: {}'.format(z8.shape))
        #print('x9.shape: {}'.format(x9.shape))
        #print('z9.shape: {}'.format(z9.shape))
        
        
        out=self.late(x9)
        #print('out.shape: {}'.format(out.shape))


        return out,[x1,x2,x3,x4,x5,x6,x7,x8,x9],[z1,z2,z3,z4,z5,z6,z7,z8,z9],[s1,s2,s3,s4]

########################## coupled_U ############################

########################## assist ############################

def Assist(fusion_band,hrmsi_band,args, init_type='kaiming', init_gain=0.02,initializer=True ):
    
    net = assist (fusion_band,hrmsi_band,args)

    return init_net(net,args.device, init_type, init_gain,initializer)

class assist(nn.Module):
    def __init__(self,fusion_band,hrmsi_band,args): 

        super().__init__()
        
        
        self.band=args.band
        self.fusion_band=fusion_band
        self.hrmsi_band=hrmsi_band
        
        self.seq1=nn.Sequential(
        nn.Conv2d(self.fusion_band,self.band,kernel_size= (5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True) nn.ReLU(inplace=True) 
                                )
        
        self.seq2=nn.Sequential(
        nn.Conv2d(self.hrmsi_band,self.band,kernel_size= (5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True) nn.ReLU(inplace=True) 
                                )
        
        self.seq3=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.band, out_channels=self.band // 5, kernel_size=1,stride=1,padding=0, bias=True), #in_channels=input_channel, out_channels=30,kernel_size=3,stride=1,padding=1
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.band // 5, out_channels=self.band, kernel_size=1,stride=1,padding=0, bias=True),
            nn.Sigmoid()
                                )
        
        
        #x7=self.ex4(torch.cat([s1,x6],dim=1))
    def forward(self,hrhsi,hrmsi):
        
        hrhsi_x1_median=self.seq1(hrhsi) #torch.Size([1, 256, 200, 200])
        
        hrmsi_Z1=self.seq2(hrmsi) #torch.Size([1, 256, 200, 200])
        
        weight=self.seq3(hrmsi_Z1) #torch.Size([1, 256, 1, 1])
        
        hrhsi_x1=weight*hrhsi_x1_median
        
        
        return hrhsi_x1, hrmsi_Z1

    
########################## assist ############################


if __name__ == "__main__":
    import numpy as np
    import os
    import scipy.io as io

    