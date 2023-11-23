from bit16to8 import Bit16to8
from model import Transpan
import torch
from config import FLAGES
from GetDataSet import GetDataSet
from torch.utils.data import DataLoader
from os.path import join
import os
import numpy as np
import scipy.io as scio
import torch.nn as nn
import time
import cv2

if not os.path.exists(FLAGES.fused_images_dir):
    os.makedirs(FLAGES.fused_images_dir)
if not os.path.exists(FLAGES.visual_img_save_path):
    os.makedirs(FLAGES.visual_img_save_path)

def get_test_set(period=None):
    dataset = GetDataSet(size=FLAGES.pan_size,source_path=FLAGES.img_path, data_path=FLAGES.data_path,
                         stride=FLAGES.stride,period=period)
    return dataset.data_generator,dataset.ms_key_list,dataset.pan_key_list

def array2mat(data_save_path,array):
    scio.savemat(data_save_path,{'imgPS':array})

def denorm(x):
    if FLAGES.if_normalize==True:
        x=x*2**11
    return x

def img_save(x,name,k):
    x = np.transpose(x, (1, 2, 0))
    if name == 'fused_images_fr':
        if not os.path.exists(join(FLAGES.fused_images_dir,'fr')):
            os.makedirs(join(FLAGES.fused_images_dir,'fr'))
        array2mat(join(FLAGES.fused_images_dir,'fr','{}.mat'.format(k)),x)
    else:
        if not os.path.exists(join(FLAGES.fused_images_dir,'rr')):
            os.makedirs(join(FLAGES.fused_images_dir,'rr'))
        array2mat(join(FLAGES.fused_images_dir,'rr','{}.mat'.format(k)),x)

transformed_testset,ms_key_list,pan_key_list= get_test_set(period='test')
testset_dataloader = DataLoader(dataset=transformed_testset, batch_size=FLAGES.test_batch_size, shuffle=False,
                                 num_workers=FLAGES.num_workers, pin_memory=True, drop_last=True)
model= Transpan().cuda()

def test():
    gen_net = join(FLAGES.backup_model_dir, '{}-{}-epochs{}.pth'.format(FLAGES.satellite,FLAGES.method, FLAGES.pth))
    # print(gen_net)
    if os.path.exists(gen_net):
        print("==> loading checkpoint '{}'".format(gen_net))
        gen_net = torch.load(gen_net)

        generator=gen_net['GenNet']
        model.load_state_dict(generator)

        print('==> loading epoch {} successful！'.format(FLAGES.pth))

    model.eval()
    with torch.no_grad():
        for p, data in enumerate(testset_dataloader):
            print(p)
            rr_ms,rr_pan,fr_ms,fr_pan= data[0].cuda(),data[1].cuda(),data[2].cuda(),data[3].cuda()

            b, c, h, w = rr_ms.shape
            s = FLAGES.pan_size // 4
            rr_ms_batch = rr_ms.reshape(b, c, h // s, s, w // s, s).permute(0, 2, 4, 1, 3, 5).reshape(
                b * h * w // s ** 2, c, s, s)

            b, c, h, w = rr_pan.shape
            s = FLAGES.pan_size
            rr_pan_batch = rr_pan.reshape(b, c, h // s, s, w // s, s).permute(0, 2, 4, 1, 3, 5).reshape(
                b * h * w // s ** 2, c, s, s)

            fused_images_rr = model(rr_ms_batch, rr_pan_batch).detach()

            b, c, h, w = fused_images_rr.shape
            s = rr_pan.shape[2] // FLAGES.pan_size
            fused_images_rr = fused_images_rr.reshape(b // s ** 2, s, s, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(
                b // s ** 2, c, s * h, s * w)
            fused_images_rr = fused_images_rr[0].cpu().numpy()
            img_save(fused_images_rr, 'fused_images_rr', int(ms_key_list[p]))

            b, c, h, w = fr_ms.shape
            s = FLAGES.pan_size // 4
            fr_ms_batch = fr_ms.reshape(b, c, h // s, s, w // s, s).permute(0, 2, 4, 1, 3, 5).reshape(
                b * h * w // s ** 2, c, s, s)

            b, c, h, w = fr_pan.shape
            s = FLAGES.pan_size
            fr_pan_batch = fr_pan.reshape(b, c, h // s, s, w // s, s).permute(0, 2, 4, 1, 3, 5).reshape(
                b * h * w // s ** 2, c, s, s)

            fused_images_fr = model(fr_ms_batch, fr_pan_batch).detach()

            b, c, h, w = fused_images_fr.shape
            s = fr_pan.shape[2] // FLAGES.pan_size
            fused_images_fr = fused_images_fr.reshape(b // s ** 2, s, s, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(
                b // s ** 2, c, s * h, s * w)
            fused_images_fr=fused_images_fr[0].cpu().numpy()
            img_save(fused_images_fr, 'fused_images_fr', int(ms_key_list[p]))


if __name__=='__main__':
    test()