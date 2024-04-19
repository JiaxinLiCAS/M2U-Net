# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:50:45 2021

@author: 13572
"""

import argparse
import torch
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
###通用参数
parser.add_argument('--scale_factor',type=int,default=12, help='缩放尺度 Houston18=8 DC=10 TG=12 Chikusei=16')
parser.add_argument('--sp_root_path',type=str, default='data/M2U-Net/spectral_response/',help='光谱相应地址')
parser.add_argument('--default_datapath',type=str, default="data/M2U-Net/",help='高光谱读取地址')
parser.add_argument('--data_name',type=str, default="TG",help='Houston18=8 DC=10 TG=12 Chikusei=16')
parser.add_argument("--gpu_ids", type=str, default='0', help='')
parser.add_argument('--checkpoints_dir',type=str, default='checkpoints',help='高光谱读取地址')
parser.add_argument('--seed',type=int, default=30,help='初始化种子')

####

parser.add_argument("--fusion_weight", type=float, default=0.5,help='default=0.5')
parser.add_argument("--band", type=int, default=260,help='文章中的P，Unet中的光谱波段数')
#parser.add_argument("--select", type=str, default=False,help='是否在第二阶段采用 退化引导融合策略')



#训练参数
#第一阶段学习率
parser.add_argument("--lr_stage1", type=float, default=0.001,help='学习率6e-3 0.001')
parser.add_argument('--niter1', type=int, default=3000, help='# 3000of iter at starting learning rate3000')
parser.add_argument('--niter_decay1', type=int, default=3000, help='# 3000of iter to linearly decay learning rate to zero3000')
#第二阶段学习率
parser.add_argument("--lr_stage2_SPe", type=float, default=4e-3,help='学习率4e-3')
parser.add_argument('--niter2_SPe', type=int, default=2000, help='#2000 of iter at starting learning rate')
parser.add_argument('--niter_decay2_SPe', type=int, default=2000, help='# 2000of iter to linearly decay learning rate to zero')
#第三阶段学习率
parser.add_argument("--lr_stage3_dip", type=float, default=4e-3,help='学习率4e-3')
parser.add_argument('--niter3_dip', type=int, default=7000, help='#7000 of iter at starting learning rate')
parser.add_argument('--niter_decay3_dip', type=int, default=7000, help='# 7000of iter to linearly decay learning rate to zero')



#添加噪声
parser.add_argument('--noise', type=str, default="No", help='Yes ,No')
parser.add_argument('--nSNR', type=int, default=35)


args=parser.parse_args()

#device = torch.device(  'cuda:{}'.format(args.gpu_ids[0])  ) if args.gpu_ids else torch.device('cpu') 
device = torch.device(  'cuda:{}'.format(args.gpu_ids)  ) if  torch.cuda.is_available() else torch.device('cpu') 
args.device=device
args.sigma = args.scale_factor / 2.35482

args.expr_dir=os.path.join('checkpoints', args.data_name+'SF'+str(args.scale_factor)+'_band'+str(args.band)+
                           '_S1_'+str(args.lr_stage1)+'_'+str(args.niter1)+'_'+str(args.niter_decay1)+
                           '_S2_'+str(args.lr_stage2_SPe)+'_'+str(args.niter2_SPe)+'_'+str(args.niter_decay2_SPe)+
                           '_S3_'+str(args.lr_stage3_dip)+'_'+str(args.niter3_dip)+'_'+str(args.niter_decay3_dip)
                           )
#opt.sigma = scale_factor / 2.35482 checkpoints_dir