'''
ddp는 어떻게 해야 하는가??

일단 간단한 ddp 학습 프로그램을 만들어 보자.
torch ddp 로 할것이다. torch 가 지원이 잘 디는거 같으니까



DDP는 N개의 processes를  spawn 해서 돌린다.

torch.cuda.set_device(i)

프로세스간 통신을 위해 torch.distributed.init_process_group이 필요한것같다.

각 모델 만들때 디바이스 아이디를 전해줘야 하는 모양.
model = DistributedDataParallel(model, device_ids=[i], output_device=i

모델 save후 load할때 map location 을 잘 해라. 아예CPU로 옮기고 저장로드하는게 낫지 않을까?

https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c 
위코드를 기준으로 작성해보자.



'''

PATH_META_CSV = '/mnt/g/Hemorrhage_IMG/dataset.csv'

# epoch
MAX_EPOCH = 200
SAMPLE_PER_EPOCH = 10*1024

# batch size 
BATCH_SIZE = 3*6 #3*3 #3의 배수 필수
IMAGE_SIZE = (256,256) #H,W
PATCH_SIZE = (16,128,128) # D,H,W
N_PATCH_PER_PATIENT = 32

# torch, cuda..
TORCH_DEVICE = 'cuda:2'  # gpu번호. 'cpu', 'cuda:0'
DATA_PARALLEL = False

# valid 
STRIDE_Z = 16
# 
STEP_PER_EPOCH = SAMPLE_PER_EPOCH//BATCH_SIZE




import argparse

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import multiprocessing as mp

import torchvision
import torchsummary
from ignite import metrics

import numpy as np
import pandas as pd
import os
import cv2
import time

from helper import imshow, cooltime
from models import vnet2
from train_shortcode import get_meta, get_data_loader_train








''' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model '''
''' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< model '''


'''===========================> train show''' 
class Average():
    def __init__(self):
        self.values = []            
        self.reset()
        return
    def update(self,list_value):
        self.values+=list_value
        return
    def compute(self):
        return np.mean(self.values)
    def reset(self):
        self.values = []   
        
def show_train_log():

    str_train = [
        f'TR {i_epoch} {i_batch}  ',
        f'R {args.local_rank}  ',
        f'L {metric_loss.compute():.5E}  ',
    ]     
    str_train = ''.join(str_train)

#     t_gpu = watch_gpu.get() / (i_batch+1)
#     t_loop = watch_loop.get() / (i_batch+1)
    
    ### train log display
    if not cooltime(key='train_show',cooltime_=10.01):
        print(str_train)

    return str_train
    
    
    

    
    
if __name__ == '__main__':
    
    ''' =================> argparse '''
    parser = argparse.ArgumentParser('DDP 3D patch seg')
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='local process rank')
    parser.add_argument('-f', help='ipython 에서 동작하게끔 f를 받음. 아무것도 안함')
    args = parser.parse_args()
    
        
        
    ''' =================> init '''
    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank==0  
    if args.is_master:
        print( "IM MASTER")
        t_start = time.time()
    
    # set the device
    args.device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(args.device)
    
    # init process gruop
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo',rank=args.local_rank)
    
    # set seed(필요하다면)
    # torrch.cuda.manual_seed(seed)
    ''' <================== init '''
    
    
    
    ''' ===================> model '''
    model = vnet2.VNetV2().to('cpu')
    if args.is_master:
        torchsummary.summary(model,PATCH_SIZE+(1,),batch_size=BATCH_SIZE,device='cpu')  # 작동 확인 차원..to(args.device)
    model.to(args.device)
    model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank)
#     if not args.is_master :
    
    
    
    ''' ===================> optimizer '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    
    
    
    ''' ====================> metric '''            
    metric_loss = Average()
    metric_macc = Average()
    
    
    ''' =====================> data '''
    ''' get meta '''
    meta_trh, meta_vlh, meta_teh, meta_trn, meta_vln, meta_ten =  get_meta(PATH_META_CSV)
#     meta_vl = pd.concat([meta_vlh,meta_vln])

    '''dataloader train'''
    dl_tr = get_data_loader_train( meta_trh, meta_trn, BATCH_SIZE, IMAGE_SIZE, N_PATCH_PER_PATIENT, PATCH_SIZE)

    
#     '''dataloader valid'''
#     # 당장은 필요 없음
#     ds_vl_patient = HemoPatientDataset(meta_vl, IMAGE_SIZE, copy_samples=1)
#     dl_vl = DataLoader( ds_vl_patient, batch_size=1, num_workers=1)

    
    

    '''================================================== train loop'''    
    i_epoch = 0         

    # iter를 꺼내놔야지 빠르다.
    iter_tr = iter(dl_tr)
    images, masks = next(iter_tr)
    

    print('train start :',args.local_rank)
    for i_epoch in range(i_epoch,MAX_EPOCH):

        ''' ================ train '''
        model.train()
        model.to(args.device)
        
        dist.barrier()
        
        for i_batch, (images, masks) in enumerate(iter_tr):

            '''----------------- 입력 준비 구간'''      
            masks_oh = torch.squeeze(masks,-1).to(torch.int64)
            masks_oh = torch.nn.functional.one_hot(masks_oh,num_classes=2).float()


            '''----------------- gpu 구간  '''

            ### weight 계산 NDHWC
            weight = torch.Tensor([1,10])

            ### 순전파~로스 
            optimizer.zero_grad() # 기존 변화도를 지우는(기존 역전파기록을 지우는)
            preds = model(images.to(args.device))
            loss = nn.BCEWithLogitsLoss(pos_weight=weight.to(args.device))(
                preds.to(args.device), masks_oh.to(args.device),
            )

            ### 역전파
            loss.backward() #이 함수는 backpropagation을 수행하여 x.grad에 변화도를 저장한다.

            ### 가중치 업데이트
            optimizer.step()
            
            ### metric
            metric_loss.update([loss.detach().cpu().numpy().tolist()])


            '''----------------- show '''
            if args.is_master: 
                show_train_log()
     
            ### exit condicion
            if i_batch >= STEP_PER_EPOCH: break;


        str_log_tr = show_train_log()

        metric_loss.reset()
