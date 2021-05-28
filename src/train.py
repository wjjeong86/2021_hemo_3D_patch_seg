'''
통채로 만들기

나중에 잘라서 파일로 만들되.
지금은 파이토치 만들어 보자.


'''


'''
###분류기(Classifier) 학습하기
https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

###custom dataset 클래스 사용.
https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html

###model train eval 모드 전환
https://wegonnamakeit.tistory.com/47
'''


MAX_EPOCH = 20
GPU_ID = '1'


import torch, torchvision, torchvision.transforms as transforms
from helper import *
import cv2, pandas as pd
from PIL import Image

from data_loader import HemoPatientDataset, HemoPatientIterableDataset, HemoRandomPatchIterableDataset


print(torch.__version__)
print('GPU:',torch.cuda.is_available())



'''============================================================ setup '''
setup_gpu(GPU_ID)



'''============================================================= data loader '''
'''
본래 dataset은 1개 이미지를 만들고 DataLoader가 배치를 만드는데 사용됨
가장 쉬운 구현을 따라서 구현해봄.

'''

##### meta csv 준비
path_hemo_csv = '/mnt/ssd0/Hemorrhage/Data/00_SNU/SN_Hemo_Data.csv'
path_normal_csv = '/mnt/ssd0/Hemorrhage/Data/00_SNU/SN_Normal_Data_v2.csv'
meta_hemo = pd.read_csv( path_hemo_csv, index_col=None)
meta_normal = pd.read_csv( path_normal_csv, index_col=None)
meta = pd.concat( [meta_hemo,meta_normal] )
meta = meta.reset_index(drop=True)

##### 경로 오류 수정
for i in meta.index:
    meta.at[i,'path_ct'] = meta.at[i,'path_ct'].replace('mnt_ssd1','ssd0')
    if str == type(meta.at[i,'path_mask']):
        meta.at[i,'path_mask'] = meta.at[i,'path_mask'].replace('mnt_ssd1','ssd0')
    else:
        meta.at[i,'path_mask'] = 'no_mask'
meta_hemo = meta[meta['is_hemo_patient']==True].reset_index(drop=True)
meta_normal = meta[meta['is_hemo_patient']==False].reset_index(drop=True)

'''========================================================= ds->ds->dl 구조'''
ds_hemo_patient = HemoPatientIterableDataset(meta_hemo)
ds_normal_patient = HemoPatientIterableDataset(meta_normal)

ds_hemo_patch = HemoRandomPatchIterableDataset(
    iter(ds_hemo_patient), mode_mask_weight=True
)
ds_normal_patch = HemoRandomPatchIterableDataset(
    iter(ds_normal_patient), mode_mask_weight=False
)

dl_hemo_patch = DataLoader( 
    ds_hemo_patch, batch_size=None, num_workers=4,
    worker_init_fn=fn_setup_seed,
)
dl_normal_patch = DataLoader(
    ds_normal_patch, batch_size=None, num_workers=4,
    worker_init_fn=fn_setup_seed,
)

# iter를 꺼내놔야지 빠르다.
iter_hemo = iter(dl_hemo_patch)
iter_normal = iter(dl_normal_patch)

for i,((_,himage5,hmask5),(_,nimage5,nmask5)) in \
    enumerate( zip(iter_hemo,iter_normal) ):

    himage = himage5[0,3,:]
    hmask = hmask5[0,3,:]
    nimage = nimage5[0,3,:]
    nmask = nmask5[0,3,:]

    summry = np.hstack( (himage,hmask,nimage,nmask))
    imshow(summry*255)

    if i>=30:
        break;

''' =========================================================== model '''

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
from torch import nn
# Define model

def conv33(Cin,Cout):
    return nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=1, padding=1, bias=False)
def bn(Cin):
    return nn.BatchNorm2d(num_features=Cin)
def relu():
    return nn.ReLU()
def pool():
    return nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

def BRC(Cin,Cout):
    return nn.Sequential(
        bn(Cin),
        relu(),
        conv33(Cin,Cout)
    )
    
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        # 32->16->8->fc
        self.entry_conv_01 = conv33(3,32)
        
        self.body_BRC_01 = BRC(32,32)
        self.body_pool_01 = pool()
        self.body_BRC_02 = BRC(32,64)
        self.body_pool_02 = pool()
        self.body_BRC_03 = BRC(64,128)
        
        self.exit_BRC_01 = BRC(128,128)
        self.exit_BRC_02 = BRC(128,128)
        
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1) # 어떻게 하든 최종 사이즈만 적으면 되나봄.
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(in_features=128,out_features=10,bias=True)
        

    def forward(self, x):
        
        x = x.permute(0,3,1,2)
        
        z = self.entry_conv_01(x)
        
        z = self.body_BRC_01(z)
        z = self.body_pool_01(z)
        z = self.body_BRC_02(z)
        z = self.body_pool_02(z)
        z = self.body_BRC_03(z)
                
        z = self.exit_BRC_01(z)
        z = self.exit_BRC_02(z)
        
        z = self.GAP(z)
        z = self.flatten(z)
        
        logits = self.fc(z)
        
        return logits

model = mymodel().to(device)
print(model)

# pred = model(image.to(device))
# pred.shape






'''================================================ loss and opt '''
loss_fn = nn.BCEWithLogitsLoss() # 원핫 인코딩 사용하기 위해서
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




'''================================================== train loop'''
i_epoch = 0 
for i_epoch in range(i_epoch,MAX_EPOCH):
    
    ''' ================ train '''
    model.train()
    for i_batch, (image, label, path) in enumerate(dl_tr):
        ### 데이터 준비
        label_oh = torch.nn.functional.one_hot(label,num_classes=10).float()


        ### 순전파~로스 
        pred = model(image.to(device))
        loss = loss_fn(pred,label_oh.to(device))

        ### 역전파
        optimizer.zero_grad() # 기존 변화도를 지우는(기존 역전파기록을 지우는)
        loss.backward() #이 함수는 backpropagation을 수행하여 x.grad에 변화도를 저장한다.
        
        ### 가중치 업데이트
        optimizer.step()
            
        ### accuracy
        acc = (pred.argmax(1)==label.to(device)).float().mean()

        ### train log display
        if cool_time(key='train_log',cooltime=1.0):
            print(
                f'TR {i_epoch}/{i_batch}   L {loss.item():>7f}   A {acc.item():.3f}'
            )
            
            
    ''' ================ validation '''
    with torch.no_grad():
        model.eval()        
        for i_batch, (image, label, path) in enumerate(dl_vl):
            ### 데이터 준비
            label_oh = torch.nn.functional.one_hot(label,num_classes=10).float()

            ### 순전파~로스
            pred = model(image.to(device))
            loss = loss_fn(pred,label_oh.to(device))
            
            ### accuracy
            acc = (pred.argmax(1)==label.to(device)).float().mean()
            

            ### display
            if cool_time(key='valid_log',cooltime=1.0):
                print(
                    f'VL {i_epoch}/{i_batch}   L {loss.item():>7f}   A {acc.item():.3f}'
                )

    
    
    
    
'''================================================== save '''

### 가중치만 저장 로드
# torch.save(model.state_dict(),'mmymodelodel.weight')
# model = mymodel()
# model.load_state_dict(torch.load('mymodel.weight'))
# model.eval()


### 모델구조와 가중치 같이 저장
torch.save(model,'mymodel.all')
model = torch.load('mymodel.all')
pred = model(image.to(device)).to('cpu')
max1 = pred.argmax(dim=1)
hit = max1==label
