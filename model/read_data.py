# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:49:02 2022

@author: 13572
"""


import torch
import os
import scipy.io as io
import numpy as np
import xlrd
#from .config import args


#from utils import util

class readdata():
    def __init__(self, args):
        #super(Dataset, self).__init__()
        self.args = args
        self.srf_gt = self.get_spectral_response(self.args.data_name) #hs_band X ms_band 的二维矩阵
        #self.msi_channel = self.srf_gt.shape[1]
        #self.hsi_channel = self.srf_gt.shape[0]
        self.psf_gt = self.matlab_style_gauss2D(shape=(self.args.scale_factor,self.args.scale_factor),sigma=self.args.sigma) #ratio X ratio 的二维矩阵
        self.sp_range = self.get_sp_range(self.srf_gt)
        data_folder = os.path.join(self.args.default_datapath, args.data_name)
        #data_folder = os.path.join(r"E:\Code\coupled\data", args.data_name)
        if os.path.exists(data_folder):
            #data_path = os.path.join(data_folder, "*.mat")
            data_path = os.path.join(data_folder, "REF.mat")
        else:
            return 0

        self.gt=io.loadmat(data_path)['REF']
        'low HSI'
        self.lr_hsi=self.generate_low_HSI(self.gt, self.args.scale_factor)
        
        'high MSI'
        self.hr_msi=self.generate_MSI(self.gt, self.srf_gt)
        
        '从msi空间降采样'
        self.lr_msi_fmsi = self.generate_low_HSI(self.hr_msi, self.args.scale_factor)
        
        '从lrhsi光谱降采样'
        self.lr_msi_fhsi= self.generate_MSI(self.lr_hsi, self.srf_gt)
     
        #测试是否相同
        #from evaluation import MetricsCal 
        #scale=10
        #sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.lr_msi_fmsi,self.lr_msi_fhsi, scale)
        #print("sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(sam,psnr,ergas,cc,rmse,Ssim,Uqi))
    
    
        #判断是否增加噪声
         
        if args.noise == 'Yes':
            
            sigmam_hsi=np.sqrt(   (self.lr_hsi**2).sum() / (10**(args.nSNR/10)) / (self.lr_hsi.size) )
            t=np.random.randn(self.lr_hsi.shape[0],self.lr_hsi.shape[1],self.lr_hsi.shape[2])
            self.lr_hsi=self.lr_hsi+sigmam_hsi*t
        
            sigmam_msi=np.sqrt(   (self.hr_msi**2).sum() / (10**(args.nSNR/10)) / (self.hr_msi.size) )
            t=np.random.randn(self.hr_msi.shape[0],self.hr_msi.shape[1],self.hr_msi.shape[2])
            self.hr_msi=self.hr_msi+sigmam_msi*t
            
        '将W×H×C的numpy转化为1×C×W×H的tensor'
      
        #self.tensor_gt = torch.from_numpy(self.gt.transpose(2,0,1).copy()).unsqueeze(0).float() 去掉了.float() ，因为.float() 会导致小数点后数字变化
        
        
        self.tensor_gt = torch.from_numpy(self.gt.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_lr_hsi = torch.from_numpy(self.lr_hsi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_hr_msi = torch.from_numpy(self.hr_msi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_lr_msi_fmsi = torch.from_numpy(self.lr_msi_fmsi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_lr_msi_fhsi = torch.from_numpy(self.lr_msi_fhsi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        
        #保存配置
        self.print_options()
        #保存真实的PSF和SRF
        self.save_psf_srf()
        #保存生成的lr_hsi和hr_msi
        self.save_lrhsi_hrmsi()
        print("readdata over")
            
    def matlab_style_gauss2D(self,shape=(3,3),sigma=2): 
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h   
            
    def get_spectral_response(self,data_name):
        xls_path = os.path.join(self.args.sp_root_path, data_name + '.xls')
        #xls_path = os.path.join(r'E:\Code\coupled\data\spectral_response', data_name + '.xls')
        if not os.path.exists(xls_path):
            raise Exception("spectral response path does not exist")
        data = xlrd.open_workbook(xls_path)
        table = data.sheets()[0]
    
        num_cols = table.ncols
        cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(0,num_cols)]
    
        sp_data = np.concatenate(cols_list, axis=1)
        sp_data = sp_data / (sp_data.sum(axis=0))
    
        return sp_data   
    
    def get_sp_range(self,srf_gt):
        HSI_bands, MSI_bands = srf_gt.shape
    
        assert(HSI_bands>MSI_bands)
        sp_range = np.zeros([MSI_bands,2])
        for i in range(0,MSI_bands):
            index_dim_0, index_dim_1 = np.where(srf_gt[:,i].reshape(-1,1)>0)
            sp_range[i,0] = index_dim_0[0] #这是索引，不是代表具体第几个波段
            sp_range[i,1] = index_dim_0[-1]
        return sp_range
    
    def downsamplePSF(self, img,sigma,stride):
        def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride,stride),sigma)
        if img.ndim == 3:
            img_w,img_h,img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w,img_h = img.shape
            img = img.reshape((img_w,img_h,1))
        from scipy import signal
        out_img = np.zeros((img_w//(stride), img_h//(stride), img_c))
        for i in range(img_c):
            out = signal.convolve2d(img[:,:,i],h,'valid')  #signal.convolve2d 要先对卷积核顺时针旋转180°
            out_img[:,:,i] = out[::stride,::stride]
        return out_img

    def generate_low_HSI(self, img, scale_factor):
        (h, w, c) = img.shape
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        return img_lr 

    def generate_MSI(self, img, srf_gt):
        w,h,c = img.shape
        self.msi_channels = srf_gt.shape[1]
        if srf_gt.shape[0] == c:
            img_msi = np.dot(img.reshape(w*h,c), srf_gt).reshape(w,h,srf_gt.shape[1])
        else:
            raise Exception("The shape of sp matrix doesnot match the image")
        return img_msi
        
    def print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.args ).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
       
        if not os.path.exists(self.args.expr_dir):
            os.makedirs(self.args.expr_dir)
            
        file_name = os.path.join(self.args.expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
    def save_psf_srf(self):
        psf_name = os.path.join(self.args.expr_dir, 'psf_gt.mat')
        srf_name = os.path.join(self.args.expr_dir, 'srf_gt.mat')
        io.savemat(psf_name,{'psf_gt': self.psf_gt})
        io.savemat(srf_name,{'srf_gt': self.srf_gt})
        
    def save_lrhsi_hrmsi(self):
        lr_hsi_name = os.path.join(self.args.expr_dir, 'lr_hsi.mat')
        hr_msi_name = os.path.join(self.args.expr_dir, 'hr_msi.mat')
        io.savemat(lr_hsi_name,{'lr_hsi': self.lr_hsi})
        io.savemat(hr_msi_name,{'hr_msi': self.hr_msi})
        io.savemat(os.path.join(self.args.expr_dir , 'gt_lr_msi.mat'), {'lr_msi_fhsi_gt': self.lr_msi_fhsi, 'lr_msi_fmsi_gt': self.lr_msi_fmsi})

        
if __name__ == "__main__":
    
    
    from config import args
    im=readdata(args)