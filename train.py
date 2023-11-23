from config import FLAGES
import time
import os
import torch
from os.path import join
from torch.utils.data import Dataset, DataLoader
from GetDataSet import GetDataSet
import torch.nn as nn
from model import Transpan
import copy
import numpy as np
import scipy.io as scio
EPS = 1e-12

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

if not os.path.exists(FLAGES.model_dir):
    os.makedirs(FLAGES.model_dir)
if not os.path.exists(FLAGES.backup_model_dir):
    os.makedirs(FLAGES.backup_model_dir)
if not os.path.exists(FLAGES.fused_images_dir):
    os.makedirs(FLAGES.fused_images_dir)
if not os.path.exists(FLAGES.visual_img_save_path):
    os.makedirs(FLAGES.visual_img_save_path)

## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calc_mse(fused,ref):
    mse=torch.mean((fused-ref)**2)
    return mse

def calc_rmse(fused,ref):
    mse=calc_mse(fused,ref)
    rmse=mse.sqrt()
    return rmse

def calc_psnr(fused,ref):
    max_val=2**11   # Bits/pixel of satellite data
    psnr=20*torch.log10(max_val/calc_rmse(fused,ref))
    return psnr

def get_data_set(period=None):
    dataset = GetDataSet(size=FLAGES.pan_size, source_path=FLAGES.img_path,
                         data_path=FLAGES.data_path,
                         stride=FLAGES.stride, period=period)
    return dataset.data_generator

transformed_trainset = get_data_set(period='train')
transformed_validset = get_data_set(period='valid')

trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=FLAGES.train_batch_size, shuffle=True,
                                 num_workers=FLAGES.num_workers, pin_memory=True, drop_last=True)
validset_dataloader = DataLoader(dataset=transformed_validset, batch_size=1)

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def train(gen_net: nn.Module, gen_optimizer, trainset_dataloader, start_epoch):
    print('===>Begin Training!')
    f=open(FLAGES.record_loss_file,'w')

    best_psnr=0
    for epoch in range(start_epoch + 1, FLAGES.total_epochs + 1):
        gen_net = gen_net.train()
        start = time.time()
        prefetcher = DataPrefetcher(trainset_dataloader)
        data = prefetcher.next()
        cnt=0
        while data is not None:
            if cnt==FLAGES.steps_per_epoch:
                break
            cnt=cnt+1

            # ---------------------
            # Train Generator
            # ---------------------
            train_ms, train_pan, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            gen_optimizer.zero_grad()
            fused= gen_net(train_ms,train_pan)

            g_loss = nn.MSELoss()(fused, label) if not FLAGES.if_l1_loss else nn.L1Loss()(fused,label)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()

            data = prefetcher.next()
        print('Training one epoch costs {} seconds.'.format(time.time()-start))
        print(
            '=> {}-{}-Epoch[{}/{}]:  g_loss: {:.5f}.'.format(
                FLAGES.satellite, FLAGES.method, epoch, FLAGES.total_epochs, g_loss.item()))
        time_epoch = (time.time() - start)
        print('==>No:epoch {} training costs {:.4f}min'.format(epoch, time_epoch / 60))

        # backup a model in time
        if epoch % FLAGES.model_backup_freq == 0:
            gen_state = {'GenNet': gen_net.state_dict(), 'epoch': epoch}
            torch.save(gen_state, join(FLAGES.backup_model_dir,'{}-{}-epochs{}.pth'.format(FLAGES.satellite, FLAGES.method, epoch)))

        start=time.time()
        gen_net = gen_net.eval()
        prefetcher = DataPrefetcher(validset_dataloader)
        data = prefetcher.next()
        psnr_avg=0.
        cnt=0
        while data is not None:
            cnt=cnt+1
            valid_rr_ms, valid_rr_pan, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            gen_optimizer.zero_grad()

            b, c, h, w = valid_rr_ms.shape
            s=FLAGES.pan_size//4
            valid_rr_ms_batch=valid_rr_ms.reshape(b,c,h//s,s,w//s,s).permute(0,2,4,1,3,5).reshape(
                b*h*w//s**2,c,s,s)

            b, c, h, w = valid_rr_pan.shape
            s = FLAGES.pan_size
            valid_rr_pan_batch=valid_rr_pan.reshape(b,c,h//s,s,w//s,s).permute(0,2,4,1,3,5).reshape(
                b*h*w//s**2,c,s,s)

            fused = gen_net(valid_rr_ms_batch, valid_rr_pan_batch).detach()

            b,c,h,w=fused.shape
            s=valid_rr_pan.shape[2]//FLAGES.pan_size
            fused=fused.reshape(b//s**2, s,s,c ,h,w).permute(0,3,1,4,2,5).reshape(
                b//s**2,c,s*h,s*w)

            psnr_avg +=calc_psnr(fused,label)
            data = prefetcher.next()

        print('Validation costs {} seconds.'.format(time.time() - start))
        print(
            '=> {}-{}-Epoch[{}/{}]:  psnr_avg: {:.5f}.'.format(
                FLAGES.satellite, FLAGES.method, epoch, FLAGES.total_epochs, psnr_avg.item()/cnt))
        f.write('{}:{:.2f}\n'.format(epoch,psnr_avg.item()/cnt))

        if psnr_avg.item()/cnt>best_psnr:
            best_psnr=psnr_avg.item()/cnt
            best_weights=copy.deepcopy({'GenNet': gen_net.state_dict(), 'best_epoch': epoch})

    f.close()
    print('best_epoch: {}, psnr: {:.2f}'.format(best_weights['best_epoch'],best_psnr))
    torch.save(best_weights, join(FLAGES.backup_model_dir,
                               '{}-{}-epochsbest.pth'.format(FLAGES.satellite, FLAGES.method)))

def main():
    total_iterations = FLAGES.total_epochs * FLAGES.steps_per_epoch
    print('total_iterations:{}'.format(total_iterations))

    gen_net = Transpan()
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     FLAGES.lr, (FLAGES.beta1, FLAGES.beta2))
    gen_net = gen_net.cuda()

    gen_net_filename = join(FLAGES.backup_model_dir,
                            '{}-{}-epochs{}.pth'.format(FLAGES.satellite, FLAGES.method, FLAGES.backup_model))
    if os.path.exists(gen_net_filename):
        print("==> loading checkpoint '{}'".format(gen_net_filename))
        gen_checkpoint = torch.load(gen_net_filename)
        gen_net.load_state_dict(gen_checkpoint['GenNet'])
        start_epoch = gen_checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')
    train(gen_net, gen_optimizer, trainset_dataloader, start_epoch)

if __name__ == '__main__':
    main()