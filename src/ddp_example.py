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

BATCH_SIZE = 32

STEP_PER_EPOCH = 10240
MAX_EPOCH = 1000


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
import os
import cv2
import time

from helper import imshow




ds_tr = torchvision.datasets.MNIST(root='data/MNIST',train=True,download=True,transform=np.float32 )
ds_te = torchvision.datasets.MNIST(root='data/MNIST',train=False,download=True,transform=np.float32)



''' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model '''
def conv2d(Cin,Cout,K=3,S=1):
    K = K + (0 if K%2==1 else 1)
    P = K//2        
    return nn.Conv2d(Cin,Cout,K,S,P,bias=False)

class BRC(nn.Module):
    def __init__(self,Cin,Cout,K=3,S=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(Cin)
        self.relu = nn.GELU()
        self.conv = conv2d(Cin,Cout,K,S)
        return
    def forward(self,Xin):
        Zout = self.conv(self.relu(self.bn(Xin)))
        return Zout
        
    
from models.mymodel import conv33,pool,bn,relu
class mymodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        #input shape : 28,28,1
        self.seq = nn.Sequential(
            conv2d(1,32,K=7,S=2), # 14,14
            BRC(32,64,K=5,S=2), # 7,7
            BRC(64,128,K=5,S=2), # 4,4
            BRC(128,128,K=3,S=1),# 4,4
            nn.Flatten(),  # 4,4,128 => 2048
            nn.BatchNorm1d(2048),
            nn.Linear(2048,10,bias=True)
        )
        return
    
    def forward(self,Xin):
        Xin = Xin.permute(0,3,1,2) # NHWC -> NCHW)
        Xin = Xin.contiguous()
        Zout = self.seq(Xin)
#         Zout = Zout.permute(0,2,3,1) # NC -> NC
        Zout = Zout.contiguous()
        return Zout
    
#model = mymodel().to('cpu')
#torchsummary.summary(model,input_size=(1,28,28),device='cpu')
''' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< model '''


if __name__ == '__main__':
    
    ''' >>>>>>>>>>>>>>>>> argparse '''
    parser = argparse.ArgumentParser('DDP example')
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='local process rank')
    parser.add_argument('-f', help='ipython 에서 동작하게끔 f를 받음. 아무것도 안함')
    args = parser.parse_args()
    
        
        
    ''' >>>>>>>>>>>>>>>>> init '''
    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank==0  
    if args.is_master:
        print( "IM MASTER")
        t_start = time.time()
    
    # set the device
    args.device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(args.device)
    print(args.device)
    
    # init process gruop
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo',rank=args.local_rank)
    
    # set seed(필요하다면)
    # torrch.cuda.manual_seed(seed)
    ''' <<<<<<<<<<<<<<<<<< init '''
    
    
    
    ''' >>>>>>>>>>>>>>>>> model '''
    model = mymodel().to(args.device)    
    model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank)
    torchsummary.summary(model,(28,28,1),device='cuda')  # 작동 확인 차원.
    
    
    
    ''' >>>>>>>>>>>>>>>>> optimizer '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    
    
    
    ''' >>>>>>>>>>>>>>>>> metric '''
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
            
    metric_loss = Average()
    metric_macc = Average()
#     metric_loss = metrics.Average()
    
    
    ''' >>>>>>>>>>>>>>>>> data '''
    ds_tr = torchvision.datasets.MNIST(root='data/MNIST',train=True,download=True,transform=np.float32 )
    dl_tr = DataLoader(
        dataset = ds_tr,
        sampler = DistributedSampler(ds_tr),
        batch_size = BATCH_SIZE,
    )
    
    ''' >>>>>>>>>>>>>>>>>>>>>>>>> epoch loop'''
    for i_epoch in range(MAX_EPOCH):
        model.train()
        model.to(args.device)
        
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
        
        ''' ============= train '''
        for i_batch, (images, labels) in enumerate(dl_tr):
            
            '''----------------- 입력 준비 구간'''      
            images = images.view(-1,28,28,1)
#             labels_oh = torch.nn.functional.one_hot(labels,num_classes=10)
               
        
            '''----------------- gpu 구간  '''            
            ### 순전파~로스 
            optimizer.zero_grad() # 기존 변화도를 지우는(기존 역전파기록을 지우는)
            preds = model(images.to(args.device))
            loss = nn.CrossEntropyLoss()(preds.to(args.device), labels.to(args.device))

            ### 역전파
            loss.backward() #이 함수는 backpropagation을 수행하여 x.grad에 변화도를 저장한다.
        
            ### 가중치 업데이트
            optimizer.step()
        
            ### metric
            if args.is_master:
                metric_loss.update([loss.detach().cpu().numpy().tolist()])
                acc = (preds.argmax(-1).to('cpu')==labels).float().numpy().tolist()
                metric_macc.update(acc)
        
            '''----------------- show '''
#             if args.is_master:
                    
#                 str_train = [
#                     f'TR {i_epoch} {i_batch}  ',
#                 ]     
#                 str_train = ''.join(str_train)
#                 print(str_train)
                
#                 #show_train_log()
            
            ### exit condicion
            if i_batch >= STEP_PER_EPOCH: break;
                
                
                
        if args.is_master:      
            
            str_train = [
                f'TR {i_epoch} {i_batch}  ',
                f'L {metric_loss.compute():.5E}  ',
                f'A {metric_macc.compute():.3f}  ',
                f'누적시간 {time.time()-t_start}   ',
            ]     
            str_train = ''.join(str_train)
            print(str_train)
            
            metric_loss.reset()
        
            
    
#             break;
#         break;
    ''' >>>>>>>>>>>>>>>>>>>>>>>>> epoch loop'''
    
    

    

    
# if __name__ == '__main__':    
    
#     N_gpus = torch.cuda.device_count()
#     world_size = N_gpus
    
#     mp.spawn(
#         main_worker,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )