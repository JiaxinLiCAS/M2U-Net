# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:03:00 2021

@author: 13572
"""

from .evaluation import MetricsCal
import os
import scipy.io as sio
import numpy as np
import hues
import torch.nn.functional as fun
import torch


    
def select_decision(Out_fhsi,Out_fmsi,blind): #Out_fhsi,Out_fmsi是四维device tensor
    
    gt=blind.gt
    
    
    Out_fhsi=Out_fhsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
    Out_fmsi=Out_fmsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
    fusion_output= Out_fhsi * blind.args.fusion_weight + Out_fmsi * (1-blind.args.fusion_weight) #numpy HWC

    ####融合后精度####
    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(gt,fusion_output, blind.args.scale_factor)
    L1=np.mean( np.abs( gt- fusion_output ))
    information="gt与fusion_output_decision\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
    print(information)
    
    
    #保存精度
    file_name = os.path.join(blind.args.expr_dir, 'Stage2.txt')
    with open(file_name, 'a') as opt_file:
        opt_file.write(information)
        opt_file.write('\n')

    #保存
    sio.savemat(os.path.join(blind.args.expr_dir, 'Out_fusion_S2.mat'), {'Out':fusion_output})


    #转为1 C H W device tensor
    fusion_output=torch.from_numpy(fusion_output.transpose(2,0,1).copy()).unsqueeze(0).float().to(blind.args.device) 
    
    
    return fusion_output
    
   
    
    
    
    

