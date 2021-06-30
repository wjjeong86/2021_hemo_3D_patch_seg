

'''
###분류기(Classifier) 학습하기
https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

###custom dataset 클래스 사용.
https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html

###model train eval 모드 전환
https://wegonnamakeit.tistory.com/47
'''


'''============================================================ setting '''

PATH_META_CSV = '/mnt/hdd0_share/2021_Hemorrhage/Hemorrhage_IMG/dataset.csv'

# epoch
MAX_EPOCH = 200
SAMPLE_PER_EPOCH = 10*1024

# batch size 
BATCH_SIZE = 3*6 #3의 배수 필수
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





'''============================================================ import '''
# torch 관련
import torch, torchvision, torchvision.transforms as transforms
from torch import nn
import torchsummary
from ignite import metrics

# data loader관련
from torch.utils.data import DataLoader
from data_loader import HemoPatientDataset

# 이외
import cv2, pandas as pd, time
from PIL import Image

# 코드 단순화 관련 
from helper import *
from train_shortcode import get_meta, get_data_loader_train
from valid_shortcode import transform_valid_input, detransform_valid_output





'''============================================================ setup '''
print(torch.__version__)
print('GPU:',torch.cuda.is_available())

# Get cpu or gpu device for training.
# device = f'cuda:{TORCH_DEVICE}' if torch.cuda.is_available() else "cpu"
device = TORCH_DEVICE
print("Using {} device".format(device))



'''============================================================= data loader '''
''' get meta '''
meta_trh, meta_vlh, meta_teh, meta_trn, meta_vln, meta_ten =  get_meta(PATH_META_CSV)
meta_vl = pd.concat([meta_vlh,meta_vln])

'''dataloader train'''
dl_tr = get_data_loader_train( meta_trh, meta_trn, BATCH_SIZE, IMAGE_SIZE, N_PATCH_PER_PATIENT, PATCH_SIZE)

'''dataloader valid'''
#####
ds_vl_patient = HemoPatientDataset(meta_vl, IMAGE_SIZE, copy_samples=1)
dl_vl = DataLoader( ds_vl_patient, batch_size=1, num_workers=1)



''' =========================================================== model '''

# import mymodel
# model= mymodel.mymodel().to('cpu')
# print(model)
# torchsummary.summary(model, PATCH_SIZE+(1,), device='cpu')

# import vnet
# model = vnet.VNet().to('cpu')
# model.apply(vnet.weights_init)

import vnet2
model = vnet2.VNetV2().to('cpu')


torchsummary.summary(model, PATCH_SIZE+(1,), device='cpu')
if DATA_PARALLEL:
    model = torch.nn.DataParallel(model)



'''================================================ loss and opt '''
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def fn_sche(step):
    lr_min,lr_max = 1e-6,1e-0
    progress = step/STEP_PER_EPOCH
    if   progress < 1.0 :  return min(lr_min+(lr_max-lr_min)*progress,lr_max)  # warmup
    elif progress < 20.0 : return lr_max / 1
    elif progress < 40.0 : return lr_max / 10
#     elif progress < 60.0 : return lr_max / 100
    else : return lr_max / 100
    
lr_sche = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda = fn_sche,
)
# lr_sche = torch.optim.lr_scheduler.CyclicLR( 
#     optimizer, base_lr=1e-6, max_lr=1e-3, 
#     step_size_up=STEP_PER_EPOCH*1, step_size_down=STEP_PER_EPOCH*4, 
#     mode='triangular',cycle_momentum=False,
# )
# lr_sche = torch.optim.lr_scheduler.OneCycleLR( 
#     optimizer=optimizer,max_lr = 1e-3, epochs = MAX_EPOCH, steps_per_epoch=STEP_PER_EPOCH,pct_start=0.1,
# )

'''=================================== metric '''
metric_loss = metrics.Average()
metric_precision = metrics.Precision()
metric_cm = metrics.ConfusionMatrix(num_classes=2)
metric_IoU = metrics.IoU(metric_cm,ignore_index=0)





'''================================================== train show'''    
def show_train_log():
    
    str_train = [
        f'TR {i_epoch} {i_batch}  ',
        f'L {metric_loss.compute():.5E}  ',
        f'IoU {metric_IoU.compute()[0]:.3f}  ',
        f'lr {lr_sche.get_last_lr()[0]:.2E}  ',
    ]     
    str_train = ''.join(str_train)
    
    ### train log display
    t_gpu = watch_gpu.get() / (i_batch+1)
    t_loop = watch_loop.get() / (i_batch+1)
    if cooltime(key='train_print',cooltime_=10.01):        
        str_print = ''.join([str_train,f'T {int(t_gpu*1000):d}/{int(t_loop*1000):d} '])
        print(str_print)
        
    if cooltime(key='train_show',cooltime_=30.0):

        pred = nn.Softmax(-1)(preds[0,4,:,:,:])
        pred = pred.detach().to('cpu').numpy()

        mask_pred = np.expand_dims(pred.argmax(-1),-1)
        pred = pred[:,:,1:2]

        image_a = images[0,3,:,:,:]
        image_b = images[0,4,:,:,:]
        image_c = images[0,5,:,:,:]
        mask = masks[0,4,:,:,:]

        summary = np.hstack((image_a,image_b,image_c,pred,mask_pred,mask))
        imshow(summary*255)
        
    return str_train

def show_valid_log():
    str_valid = [
        f'VL {i_epoch} {i_batch}  ',
        f'L {metric_loss.compute():.5E}  ',
        f'IoU {metric_IoU.compute()[0]:.3f}  ',
    ]            
    str_valid = ''.join(str_valid)
    
    t_gpu = watch_gpu.get() / (i_batch+1)
    t_loop = watch_loop.get() / (i_batch+1)
    if cooltime(key='valid_print',cooltime_=10.0):
        str_print = ''.join([str_valid, f'T {int(t_gpu*1000):d}/{int(t_loop*1000):d} '])
        print( str_print)
        
    return str_valid
        

def to_cl(tensor):
    if   tensor.dim() == 5: return tensor.permute(0,2,3,4,1)
    elif tensor.dim() == 4: return tensor.permute(0,2,3,1)
    else:                   assert not f'unimplemented for {tensor.dim()} dims tensor'
        
def to_cf(tensor):
    if   tensor.dim() == 5: return tensor.permute(0,4,1,2,3)
    elif tensor.dim() == 4: return tensor.permute(0,3,1,2)
    else:                   assert not f'unimplemented for {tensor.dim()} dims tensor'

        
        
'''================================================== save 준비'''
os.makedirs('../train_log',exist_ok=True)
torch.save(model,'../train_log/weight_init.torch_model_save') #초기상태저장



        
'''================================================== train loop'''    
i_epoch = 0         
loss_vl_max = 999999.0
iou_vl_max = 0.0


# iter를 꺼내놔야지 빠르다.
iter_tr = iter(dl_tr)
images, masks = next(iter_tr)

watch_loop = StopWatch()
watch_gpu = StopWatch()


for i_epoch in range(i_epoch,MAX_EPOCH):
    
    ''' ================ train '''
    watch_loop.reset()
    watch_gpu.reset()
    watch_loop.start()
    model.train()
    model.to(device)
    for i_batch, (images, masks) in enumerate(iter_tr):
        
        '''----------------- 입력 준비 구간'''      
        masks_oh = torch.squeeze(masks,-1).to(torch.int64)
        masks_oh = torch.nn.functional.one_hot(masks_oh,num_classes=2).float()
               
        
        '''----------------- gpu 구간  '''
        watch_gpu.start()
        
        ### weight 계산 NDHWC
        weight = torch.Tensor([1,10])
            
        ### 순전파~로스 
        optimizer.zero_grad() # 기존 변화도를 지우는(기존 역전파기록을 지우는)
        preds = model(images.to(device))
        losses = nn.BCEWithLogitsLoss(pos_weight=weight.to(device))(
            preds.to(device), masks_oh.to(device),
        )

        ### 역전파
        losses.backward() #이 함수는 backpropagation을 수행하여 x.grad에 변화도를 저장한다.
        
        ### 가중치 업데이트
        optimizer.step()
        lr_sche.step()
        
        ### metric
        metric_loss.update(losses.to('cpu').to(torch.float64))
        metric_cm.update((to_cf(preds),masks_oh.argmax(-1).to(device)))
        
        watch_gpu.stop()
        watch_loop.stop()
        
        '''----------------- show '''
        show_train_log()
        
        watch_loop.start()        
        ### exit condicion
        if i_batch >= STEP_PER_EPOCH: break;
            
            
    str_log_tr = show_train_log()
    
    metric_loss.reset()
    metric_cm.reset()
            
            
    ''' ================ validation '''
    ''' 한 사람 통채로 해야 될텐데...
    '''
    with torch.no_grad():
        watch_loop.reset()
        watch_gpu.reset()
        watch_loop.start()
        model.eval()
        model.to(device)         
        for i_batch, (_, image, mask, idx_meta) in enumerate(dl_vl):        
            
            Dorg = image.shape[1]
            image, mask, mask_oh, pred, pred_count = transform_valid_input(image,mask,STRIDE_Z)
            
            ''' 각 depth마다.. '''
            watch_gpu.start()
            for i_depth in range(0,image.shape[1]-PATCH_SIZE[0]+1,STRIDE_Z):
                
                sub_i = image[0:1,i_depth:i_depth+PATCH_SIZE[0]]                    
                
                ### 순전파
                sub_p = model(sub_i.to(device))            
                
                ### 결과 합치기
                pred[:,i_depth:i_depth+PATCH_SIZE[0]] += sub_p.to('cpu')
                pred_count[:,i_depth:i_depth+PATCH_SIZE[0]] += torch.ones_like(sub_p, device='cpu')
            
            
            
            image, mask, mask_oh, pred = detransform_valid_output(image,mask,mask_oh,pred,pred_count,Dorg)
            
            ''' loss, metric'''
            ### weight 계산 NDHWC
            alpha = 0.99
            weight = torch.Tensor([1,10])
            
            loss = nn.BCEWithLogitsLoss(pos_weight=weight)(pred,mask_oh)
                       
            ### metric
            metric_loss.update(loss)
            metric_cm.update((to_cf(pred),mask_oh.argmax(-1)))            
                # confusion matrix update( (y_pred,y))
                # y_pred : (batch_size, num_classes, ...)
                # y : (batch_sizes, ...)
                             
                    
            watch_gpu.stop()
            watch_loop.stop()

            show_valid_log()

            watch_loop.start()
            
            
    str_log_vl = show_valid_log()
    loss_vl = metric_loss.compute()
    iou_vl = metric_IoU.compute()
    metric_loss.reset()
    metric_cm.reset()
    
    
    ''' ================ save '''
    ''' loss가 최저일때 세이브.
    '''
    os.makedirs('../train_log',exist_ok=True)
    flag_best_loss, flag_best_iou = '',''
    
    model.to('cpu')
    torch.save(model,'../train_log/weight_last.torch_model_save')
    
    if loss_vl < loss_vl_max:
        loss_vl_max = loss_vl
        torch.save(model,'../train_log/weight_best_loss.torch_model_save')
        flag_best_loss = '*'

    if iou_vl_max < iou_vl:
        iou_vl_max = iou_vl
        torch.save(model,'../train_log/weight_best_iou.torch_model_save')
        flag_best_iou = '!'
                        
    with open('../train_log/train_log.txt','a+') as f:
        f.write( str_log_tr+'   ') 
        f.write( str_log_vl+'   ')
        f.write( flag_best_loss)
        f.write( flag_best_iou)            
        f.write('\n')