# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:06:08 2021

@author: 13572
"""

#说明l:输出的是W H C大小的numpy格式数据
import numpy as np

import os
import scipy.io as io


def compute_sam(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-7)
    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    
    return mSAM

def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim ==3

    img_w, img_h, img_c = x_true.shape
    ref = x_true.reshape(-1, img_c)
    #print(ref)
    tar = x_pred.reshape(-1, img_c)
    msr = np.mean((ref - tar)**2, 0) #列和
    #print(msr)
    max2 = np.max(ref,0)**2
    #print(max2)
    psnrall = 10*np.log10(max2/msr)
    #print(psnrall)
    m_psnr = np.mean(psnrall)
    #print(m_psnr)
    psnr_all = psnrall.reshape(img_c)
    #print( psnr_all)
    return m_psnr


def compute_ergas(x_true, x_pred, scale_factor):
    assert x_true.ndim == 3 and x_pred.ndim ==3 and x_true.shape == x_pred.shape
    
    img_w, img_h, img_c = x_true.shape

    err = x_true - x_pred
    ERGAS = 0
    for i in range(img_c):
        ERGAS = ERGAS + np.mean(  err[:,:,i] **2 / np.mean(x_true[:,:,i]) ** 2  )
    
    ERGAS = (100 / scale_factor) * np.sqrt((1/img_c) * ERGAS)
    return ERGAS


def compute_cc(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    result=np.ones((img_c,))
    for i in range(0,img_c):
        CCi=np.corrcoef(x_true[:,:,i].flatten(),x_pred[:,:,i].flatten())
        result[i]=CCi[0,1]
        #print('result[i]',result[i])
        #print('CCi',CCi)
    #print(result)
    return result.mean()

def compute_rmse(x_true, x_pre):
     img_w, img_h, img_c = x_true.shape
     return np.sqrt(  ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c)   )
    


      
def MetricsCal(x_true,x_pred, scale):# c,w,h

    sam=compute_sam(x_true, x_pred)
    
    psnr=compute_psnr(x_true, x_pred)
    
    ergas=compute_ergas(x_true, x_pred, scale)
    
    cc=compute_cc(x_true, x_pred)
    
    rmse=compute_rmse(x_true, x_pred)
    

    
    from skimage.metrics import structural_similarity as ssim
    ssims = []
    for i in range(x_true.shape[2]):
        ssimi = ssim(x_true[:,:,i], x_pred[:,:,i], data_range=x_pred[:,:,i].max() - x_pred[:,:,i].min())
        ssims.append(ssimi)
    Ssim = np.mean(ssims)
    
    from sewar.full_ref import uqi
    Uqi= uqi(x_true,  x_pred)
    

    return sam,psnr,ergas,cc,rmse,Ssim,Uqi
    


if __name__ == "__main__":
    pass
    
