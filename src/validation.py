'''
학습된 모델 로드해서 성능 확인해 보기
'''


PATH_MODEL = '/mnt/hdd2_member/wjjeong/03.project/2021_hemo_3D_patch_seg/train_logs/210625_vnet2/train_log/weight_best_loss.torch_model_save'
PATH_META_CSV = '/mnt/hdd0_share/2021_Hemorrhage/Hemorrhage_IMG/dataset.csv'

IMAGE_SIZE = (256,256) #H,W
PATCH_SIZE = (16,128,128) # D,H,W
TORCH_DEVICE = 'cuda:0'  # gpu번호 'cuda:0' or 'cpu'

STRIDE_Z = 8



# torch 관련
import torch, torchvision, torchvision.transforms as transforms
import torchsummary
from torch import nn
from ignite import metrics

# data loader관련
from torch.utils.data import DataLoader
from data_loader import HemoPatientDataset

# 이외
import cv2, pandas as pd, time
from sklearn.metrics import jaccard_score, precision_score, recall_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

# 코드 단순화 관련 만든 것들
from helper import *
from train_shortcode import get_meta
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
_, _, meta_teh, _, _, meta_ten =  get_meta(PATH_META_CSV)
meta_te = pd.concat([meta_teh,meta_ten])


'''dataloader valid'''
#####
ds_te_patient = HemoPatientDataset(meta_te, IMAGE_SIZE, copy_samples=1)
dl_te = DataLoader( ds_te_patient, batch_size=1, num_workers=4)
        
# dl_te = DataLoader(ds_te_pat)

#################################################################################

''' =========================================================== model '''
model = torch.load(PATH_MODEL, map_location='cpu')
# torchsummary.summary(model, PATCH_SIZE+(1,), device='cpu')





'''================================================ loss and opt '''
metric_loss = metrics.Average()
metric_precision = metrics.Precision()
metric_cm = metrics.ConfusionMatrix(num_classes=2)
metric_IoU = metrics.IoU(metric_cm,ignore_index=0)




'''================================================== train show'''  
def valid_show():
    str_valid = [
        f'VL {i_batch}  ',
        f'L {metric_loss.compute():>7f} ',
        f'IoU {metric_IoU.compute()[0]:.3f} ',
    ]            
    str_valid = ''.join(str_valid)
    
    if cooltime(key='valid_print',cooltime_=0.01):
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

        
        
'''================================================== valid loop'''    
t_loop = time.time()
meta_save = meta_te.copy()

''' ================ validation '''
''' 한 사람 통채로 해야 될텐데...
'''
with torch.no_grad():
    model.eval()
    model.to(device)

    for i_batch, (_, image, mask, idx_meta) in enumerate(dl_te): 
        
        Dorg = image.shape[1]
        idx_meta = [int(m) for m in idx_meta]
        meta_patient = meta_te.loc[idx_meta]
        
        image, mask, mask_oh, pred, pred_count = transform_valid_input(image,mask,STRIDE_Z)

        ''' 각 depth마다.. '''
        t_gpu = time.time()
        for i_depth in range(0,image.shape[1]-PATCH_SIZE[0]+1,STRIDE_Z):

            sub_i = image[0:1,i_depth:i_depth+PATCH_SIZE[0]]                    

            ### 순전파
            sub_p = model(sub_i.to(device))            

            ### 결과 합치기
            pred      [:,i_depth:i_depth+PATCH_SIZE[0]] += sub_p.to('cpu')
            pred_count[:,i_depth:i_depth+PATCH_SIZE[0]] += torch.ones_like(sub_p, device='cpu')
            
            
            
        image, mask, mask_oh, pred = detransform_valid_output(image,mask,mask_oh,pred,pred_count,Dorg)

        ''' loss, metric'''
        ### weight 계산 NDHWC
        weight_pos = mask     / (     mask.sum(dim=[1,2,3,4],keepdim=True) * 1.0 +1e-10)
        weight_neg = (1-mask) / ( (1-mask).sum(dim=[1,2,3,4],keepdim=True) * 1.0 +1e-10)
        weight = (weight_pos + weight_neg) * weight_pos.numel()

        loss = nn.BCEWithLogitsLoss(weight)(pred,mask_oh)

        ### metric
        metric_loss.update(loss)
        metric_cm.update((to_cf(pred),mask_oh.argmax(-1)))            
            # confusion matrix update( (y_pred,y))
            # y_pred : (batch_size, num_classes, ...)
            # y : (batch_sizes, ...)
        


        
        ''' 결과 display & save '''
        ### display
        t_gpu = time.time()-t_gpu        
        t_loop = time.time()-t_loop
        
        valid_show()
        
        t_loop = time.time()
        
        ### save        
        os.makedirs('../valid_log/{}',exist_ok=True)
          
        images_pred_pos = pred.softmax(-1)[0,:,:,:,1].numpy()
        images_pred_mask = images_pred_pos>0.5
        images_input = image[0,:,:,:,0].numpy()
        images_mask = mask_oh[0,:,:,:,1].numpy()
        
        img_stack = []
        for idx,ipp,ipm,ii,im in zip( idx_meta,images_pred_pos,images_pred_mask, images_input,images_mask ):
            img_hstack = np.hstack( [ii,ipp,ipm,im] )
            img_stack.append( img_hstack)
                        
        for idx,ipp,ipm,im in zip( idx_meta,images_pred_pos,images_pred_mask,images_mask):
            iou = jaccard_score( y_true=im.flatten(),y_pred=ipm.flatten(),zero_division=0.0 )
            
            meta_save.at[idx,'loss'] = float(loss)
            meta_save.at[idx,'n_pixels_mask'] = int(im.sum())
            meta_save.at[idx,'n_pixels_pred_mask'] = int(ipm.sum())            
            meta_save.at[idx,'prob_max'] = ipp.max()
            meta_save.at[idx,'iou'] = float(iou)
        
        meta_save.to_csv(f'../valid_log/result.csv',index=False)
        
        img_stack = np.vstack(img_stack)
        path_image_save = f'../valid_log/{meta_save.at[idx,"id_patient"]}.jpg'
        cv2.imwrite( path_image_save, img_stack*255)
        



str_log_vl = valid_show()
print(str_log_vl)
loss_vl = metric_loss.compute()



### slice 단위 prec, recal, accuracy
y_pred = list(meta_save['n_pixels_pred_mask'] >0)
y_true = list(meta_save['n_pixels_mask']>0)
prob_max = list(meta_save['prob_max'])

prec = precision_score(y_true=y_true,y_pred=y_pred)
recall = recall_score(y_true=y_true,y_pred=y_pred)
pr_curve = precision_recall_curve(y_true=y_true,probas_pred=prob_max)
ap = average_precision_score(y_true=y_true,y_score=prob_max)
# fpr, tpr, _ = roc_curve( y_true=y_true, y_score=prob_max)
# auc = roc_auc_score( y_true=y_true, y_score=prob_max)


import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))
plt.plot(pr_curve[1],pr_curve[0])
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)
plt.legend([f'AP={ap:.5f}'],loc='best',fontsize=12)
plt.show()
print(f'prec {prec:.3f}   recall {recall:.3f}')


### 환자단위.
list_id_patient = np.unique(meta_save['id_patient'])
prob_max, y_true = [],[]
for id_patient in list_id_patient:
    meta_patient = meta_save[meta_save['id_patient']==id_patient]
    prob_max.append( max(meta_patient['prob_max']))
    y_true.append( int(meta_patient['is_hemo_patient'].iloc[0]==1))
y_pred = [pm>0.5 for pm in prob_max]


prec = precision_score(y_true=y_true,y_pred=y_pred)
recall = recall_score(y_true=y_true,y_pred=y_pred)
pr_curve = precision_recall_curve(y_true=y_true,probas_pred=prob_max)
ap = average_precision_score(y_true=y_true,y_score=prob_max)
# fpr, tpr, _ = roc_curve( y_true=y_true, y_score=prob_max)
# auc = roc_auc_score( y_true=y_true, y_score=prob_max)


import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))
plt.plot(pr_curve[1],pr_curve[0])
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)
plt.legend([f'AP={ap:.5f}'],loc='best',fontsize=12)
plt.show()
print(f'prec {prec:.3f}   recall {recall:.3f}')
