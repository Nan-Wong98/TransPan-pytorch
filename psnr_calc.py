import torch

def calc_mse(fused,ref):
    mse=torch.mean((fused-ref)**2)
    return mse

def calc_rmse(fused,ref):
    mse=calc_mse(fused,ref)
    rmse=mse.sqrt()
    return rmse

def calc_psnr(fused,ref):
    max_val=2**11   # Bits/pixel of satellite data
    psnr=20*torch.log10(max_val*1./calc_rmse(fused,ref))
    return psnr