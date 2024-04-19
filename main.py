# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:48:00 2022

@author: 13572
"""

#❗❗❗❗❗❗❗❗❗我的微信是:BatAug,欢迎各位同行交流合作 ❗❗❗❗❗❗❗❗❗❗❗❗❗
#❗❗❗❗❗❗❗❗❗我的微信是:BatAug,欢迎各位同行交流合作 ❗❗❗❗❗❗❗❗❗❗❗❗❗
import torch
import torch.nn as nn
import time
import numpy as np
import hues
import os
import random
import scipy.io as sio
from model.config import args
from model.evaluation import MetricsCal
from model.select import select_decision
import time
 

print('我的微信是:BatAug,欢迎各位同行交流合作')
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
        
setup_seed(args.seed) 


#第一阶段
from model.srf_psf_layer import Blind       #将退化函数看作一层的参数
blind = Blind(args)

start = time.perf_counter() # 记录开始时间

lr_msi_fhsi_est, lr_msi_fmsi_est=blind.train() #device格式的1 C H W张量
blind.get_save_result() #保存PSF SRF
psf = blind.model.psf.data.cpu().detach().numpy()[0,0,:,:] #15 x 15  numpy
srf = blind.model.srf.data.cpu().detach().numpy()[:,:,0,0].T #46 x 8   numpy
psf_gt=blind.psf_gt #15 x 15   numpy
srf_gt=blind.srf_gt  #46 x 8   numpy

end = time.perf_counter()   # 记录结束时间
elapsed_S1 = end - start        # 计算经过的时间（单位为秒）

#第二阶段
from model.spectral_up import spectral_SR

spectral_sr=spectral_SR(args,lr_msi_fhsi_est.clone().detach(), lr_msi_fmsi_est.clone().detach(),  #lr_msi_fhsi_est, lr_msi_fmsi_est
                           blind.tensor_lr_hsi,blind.tensor_hr_msi,blind.gt) 
start = time.perf_counter() # 记录开始时间
Out_fhsi,Out_fmsi=spectral_sr.train() #返回的是四维tensor device上
end = time.perf_counter()   # 记录结束时间
elapsed_S2 = end - start        # 计算经过的时间（单位为秒）    

fusion_output=select_decision(Out_fhsi,Out_fmsi,blind) #返回的是四维tensor device上
 

#第三阶段
from model.dip import dip 
DIP=dip(args,fusion_output,psf,srf,blind)
start = time.perf_counter() # 记录开始时间  
out=DIP.train()
end = time.perf_counter()   # 记录结束时间
elapsed_S3 = end - start        # 计算经过的时间（单位为秒） 


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(blind.model))
print(get_parameter_number(spectral_sr.two_stream))
print(get_parameter_number(DIP.net))



print(elapsed_S1,elapsed_S2,elapsed_S3)

print('我的微信是:BatAug,欢迎各位同行交流合作')



