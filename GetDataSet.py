import numpy as np
import os
import h5py
# import gdal
import scipy.io as scio
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from config import FLAGES

class GetDataSet(object):
    def __init__(self, size, source_path, data_path, stride, period='train'):
        self.size = size
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_save_path=data_path+FLAGES.train_img_filename
        if not os.path.exists(data_save_path):
            self.make_data(source_path, data_save_path, size,stride)
        # self.train,self.label,self.test_fr,self.test_rr,self.ms_origin=None,None,None,None,None
        if period=='train':
            self.train_ms, self.train_pan, self.train_label = self.read_data(data_save_path, period=period)
        elif period=='valid':
            self.valid_rr_ms, self.valid_rr_pan,self.valid_fr_ms, self.valid_fr_pan=\
                self.read_data(data_save_path, period=period)
        else:
            self.test_rr_ms, self.test_rr_pan,self.test_fr_ms, self.test_fr_pan,self.ms_key_list,\
            self.pan_key_list=self.read_data(data_save_path, period=period)
        self.data_generator = self.generator(period)

    def generator(self,period=None):
        if period=='train':
            dataset = TrainsetFromFolder(self.train_ms, self.train_pan, self.train_label,transform=transforms.Compose([ToTensor()]))
        elif period=='valid':
            dataset = ValidsetFromFolder(self.valid_fr_ms, self.valid_fr_pan, self.valid_rr_ms, self.valid_rr_pan,
                                         transform=transforms.Compose([ToTensor()]))
        else:
            dataset = TestsetFromFolder(self.test_fr_ms, self.test_fr_pan, self.test_rr_ms, self.test_rr_pan,
                                         transform=transforms.Compose([ToTensor()]))
        return dataset

    def read_data(self, path, period):
        f = h5py.File(path, 'r')
        if period == 'train':
            train_ms = np.array(f['train_ms'])
            train_pan = np.array(f['train_pan'])
            train_label = np.array(f['train_label'])
            return train_ms,train_pan,train_label
        elif period=='valid':
            valid_fr_ms = np.array(f['valid_fr_ms'])
            valid_fr_pan = np.array(f['valid_fr_pan'])
            valid_rr_ms = np.array(f['valid_rr_ms'])
            valid_rr_pan = np.array(f['valid_rr_pan'])
            return valid_rr_ms,valid_rr_pan,valid_fr_ms,valid_fr_pan
        else:
            test_fr_ms = np.array(f['test_fr_ms'])
            test_fr_pan = np.array(f['test_fr_pan'])
            test_rr_ms = np.array(f['test_rr_ms'])
            test_rr_pan = np.array(f['test_rr_pan'])
            ms_key_list=f['ms_key_list']
            pan_key_list=f['pan_key_list']
            return test_rr_ms,test_rr_pan,test_fr_ms,test_fr_pan,ms_key_list,pan_key_list

    def make_data(self, source_path, data_save_path, size,stride):
        ########################################
        # Preparation of train_set
        ########################################
        train_ms_files = glob.glob(source_path + "\\train\\ms_256\\*.mat")
        train_pan_files = glob.glob(source_path + "\\train\\pan_1024\\*.mat")
        train_ms_list = []
        train_pan_list = []
        train_label_list = []
        for file in range(len(train_ms_files)):
            train_ms = self.read_img(train_ms_files[file], 'ms')
            train_pan = self.read_img(train_pan_files[file], 'pan')
            train_label=train_ms
            train_ms = cv2.resize(train_ms, (train_ms.shape[0] // 4, train_ms.shape[1] // 4),cv2.INTER_CUBIC)
            train_pan = cv2.resize(train_pan, (train_pan.shape[0] // 4, train_pan.shape[1] // 4),cv2.INTER_CUBIC)

            train_ms_patch = self.crop_to_patch(train_ms, size//4,stride//4)
            train_pan_patch = self.crop_to_patch(train_pan[:,:,np.newaxis], size,stride)
            train_label_patch = self.crop_to_patch(train_label, size,stride)

            train_ms_list.append(train_ms_patch)
            train_pan_list.append(train_pan_patch)
            train_label_list.append(train_label_patch)
        train_ms_list = np.concatenate(train_ms_list, axis=0)
        train_pan_list = np.concatenate(train_pan_list, axis=0)
        train_label_list = np.concatenate(train_label_list, axis=0)
        print('The number of train patch is: ' + str(len(train_ms_list)))

        f = h5py.File(data_save_path, 'w')
        f.create_dataset('train_ms', data=train_ms_list)
        f.create_dataset('train_pan', data=train_pan_list)
        f.create_dataset('train_label', data=train_label_list)
        # f.close()

        ########################################
        # Preparation of valid_set
        ########################################
        valid_ms_files = glob.glob(source_path + "\\valid\\ms_256\\*.mat")
        valid_pan_files = glob.glob(source_path + "\\valid\\pan_1024\\*.mat")
        valid_fr_ms_list = []
        valid_fr_pan_list=[]
        valid_rr_ms_list = []
        valid_rr_pan_list = []
        for file in range(len(valid_ms_files)):
            valid_fr_ms = self.read_img(valid_ms_files[file], 'ms')
            valid_fr_pan = self.read_img(valid_pan_files[file], 'pan')
            valid_rr_ms = cv2.resize(valid_fr_ms,(valid_fr_ms.shape[0]//4,valid_fr_ms.shape[1]//4),cv2.INTER_CUBIC)
            valid_rr_pan = cv2.resize(valid_fr_pan,(valid_fr_pan.shape[0]//4,valid_fr_pan.shape[1]//4),cv2.INTER_CUBIC)

            valid_fr_ms=valid_fr_ms[np.newaxis,:,:,:]
            valid_fr_ms_list.append(valid_fr_ms)

            valid_rr_ms = valid_rr_ms[np.newaxis, :, :, :]
            valid_rr_ms_list.append(valid_rr_ms)

            valid_fr_pan = valid_fr_pan[np.newaxis, :, :, np.newaxis]
            valid_fr_pan_list.append(valid_fr_pan)

            valid_rr_pan = valid_rr_pan[np.newaxis, :, :, np.newaxis]
            valid_rr_pan_list.append(valid_rr_pan)

        valid_fr_ms_list=np.concatenate(valid_fr_ms_list,axis=0)
        valid_fr_pan_list = np.concatenate(valid_fr_pan_list, axis=0)
        valid_rr_ms_list = np.concatenate(valid_rr_ms_list, axis=0)
        valid_rr_pan_list = np.concatenate(valid_rr_pan_list, axis=0)
        print('The number of valid patch is: ' + str(len(valid_fr_ms_list)))

        # f = h5py.File(data_save_path, 'w')
        f.create_dataset('valid_fr_ms', data=valid_fr_ms_list)
        f.create_dataset('valid_fr_pan', data=valid_fr_pan_list)
        f.create_dataset('valid_rr_ms', data=valid_rr_ms_list)
        f.create_dataset('valid_rr_pan', data=valid_rr_pan_list)
        # f.close()

        ########################################
        # Preparation of test_set
        ########################################
        test_ms_files= glob.glob(source_path + "\\test\\ms_256\\*.mat")
        test_pan_files = glob.glob(source_path + "\\test\\pan_1024\\*.mat")
        test_fr_ms_list = []
        test_fr_pan_list = []
        test_rr_ms_list = []
        test_rr_pan_list = []
        source_ms_key_list=[]
        source_pan_key_list = []
        for file in range(len(test_ms_files)):
            source_ms_key=test_ms_files[file].split("\\")[-1].split(".")[0]
            source_ms_key_list.append(source_ms_key.encode())

            source_pan_key = test_pan_files[file].split("\\")[-1].split(".")[0]
            source_pan_key_list.append(source_pan_key.encode())

            test_fr_ms = self.read_img(test_ms_files[file], 'ms')
            test_fr_pan = self.read_img(test_pan_files[file], 'pan')

            test_rr_ms=cv2.resize(test_fr_ms,(test_fr_ms.shape[0]//4,test_fr_ms.shape[1]//4),cv2.INTER_CUBIC)
            test_rr_pan = cv2.resize(test_fr_pan, (test_fr_pan.shape[0] // 4, test_fr_pan.shape[1] // 4), cv2.INTER_CUBIC)

            test_fr_ms = test_fr_ms[np.newaxis, :, :, :]
            test_fr_ms_list.append(test_fr_ms)

            test_rr_ms = test_rr_ms[np.newaxis, :, :, :]
            test_rr_ms_list.append(test_rr_ms)

            test_fr_pan = test_fr_pan[np.newaxis, :, :, np.newaxis]
            test_fr_pan_list.append(test_fr_pan)

            test_rr_pan = test_rr_pan[np.newaxis, :, :, np.newaxis]
            test_rr_pan_list.append(test_rr_pan)

        test_fr_ms_list = np.concatenate(test_fr_ms_list, axis=0)
        test_fr_pan_list = np.concatenate(test_fr_pan_list, axis=0)
        test_rr_ms_list = np.concatenate(test_rr_ms_list, axis=0)
        test_rr_pan_list = np.concatenate(test_rr_pan_list, axis=0)
        print('The number of test patch is: ' + str(len(test_fr_ms_list)))

        # f = h5py.File(data_save_path, 'w')
        f.create_dataset('test_fr_ms', data=test_fr_ms_list)
        f.create_dataset('test_fr_pan', data=test_fr_pan_list)
        f.create_dataset('test_rr_ms', data=test_rr_ms_list)
        f.create_dataset('test_rr_pan', data=test_rr_pan_list)
        f.create_dataset('ms_key_list', data=source_ms_key_list)
        f.create_dataset('pan_key_list', data=source_pan_key_list)
        f.close()

    def crop_to_patch(self, img,size, stride):
        h = img.shape[0]
        w = img.shape[1]
        all_img = []
        for i in range(0, h,stride):
            for j in range(0, w,stride):
                if i + size <= h and j + size <= w:
                    patch = img[i:i + size, j:j + size, :]
                    all_img.append(patch)
        return all_img

    def read_img(self, path, name):
        if name == 'ms':
            img = scio.loadmat(path)['imgMS']
        else:
            img = scio.loadmat(path)['imgPAN']
        return img

class TrainsetFromFolder(Dataset):
    def __init__(self, train_ms,train_pan,train_label, transform=None):
        self.train_ms=train_ms
        self.train_pan = train_pan
        self.train_label=train_label
        self.transform = transform

    def __len__(self):
        return len(self.train_ms)

    def __getitem__(self, index):   # idx的范围是从0到len（self）
        ms=self.train_ms[index]
        pan=self.train_pan[index]
        label=self.train_label[index]
        if self.transform:
            ms = self.transform(ms)
            pan = self.transform(pan)
            label = self.transform(label)
            return ms,pan,label

class ValidsetFromFolder(Dataset):
    def __init__(self, valid_fr_ms,valid_fr_pan,valid_rr_ms,valid_rr_pan, transform=None):
        self.valid_fr_ms=valid_fr_ms
        self.valid_fr_pan = valid_fr_pan
        self.valid_rr_ms=valid_rr_ms
        self.valid_rr_pan = valid_rr_pan
        self.transform = transform

    def __len__(self):
        return len(self.valid_fr_ms)

    def __getitem__(self, index):   # idx的范围是从0到len（self）
        rr_ms=self.valid_rr_ms[index]
        rr_pan=self.valid_rr_pan[index]
        fr_ms = self.valid_fr_ms[index]
        fr_pan = self.valid_fr_pan[index]
        if self.transform:
            rr_ms = self.transform(rr_ms)
            rr_pan = self.transform(rr_pan)
            fr_ms = self.transform(fr_ms)
            fr_pan = self.transform(fr_pan)
            return rr_ms,rr_pan,fr_ms,fr_pan

class TestsetFromFolder(Dataset):
    def __init__(self, test_fr_ms,test_fr_pan,test_rr_ms,test_rr_pan, transform=None):
        self.test_fr_ms=test_fr_ms
        self.test_fr_pan = test_fr_pan
        self.test_rr_ms=test_rr_ms
        self.test_rr_pan = test_rr_pan
        self.transform = transform

    def __len__(self):
        return len(self.test_rr_pan)

    def __getitem__(self, index):   # idx的范围是从0到len（self）
        rr_ms=self.test_rr_ms[index]
        rr_pan=self.test_rr_pan[index]
        fr_ms = self.test_fr_ms[index]
        fr_pan = self.test_fr_pan[index]
        if self.transform:
            rr_ms = self.transform(rr_ms)
            rr_pan = self.transform(rr_pan)
            fr_ms = self.transform(fr_ms)
            fr_pan = self.transform(fr_pan)
            return rr_ms,rr_pan,fr_ms,fr_pan

class ToTensor(object):
    def __call__(self, input):
        input=input*1.0
        input = np.transpose(input, (2, 0, 1))
        input = torch.from_numpy(input).type(torch.FloatTensor)
        return input







