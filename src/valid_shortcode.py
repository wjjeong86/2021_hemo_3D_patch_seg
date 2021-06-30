import numpy as np

import torch
                
def transform_valid_input(image,mask,stride_z):

    ''' k의배수로 depth를 조절 '''
    Dorg = image.shape[1]
    k = stride_z
    Dnew = Dorg//k*k + (Dorg%k>0)*k
    Dpad = Dnew-Dorg

    image = np.pad(image, ((0,0),(0,Dpad),(0,0),(0,0),(0,0)) )
    mask  = np.pad(mask,  ((0,0),(0,Dpad),(0,0),(0,0),(0,0)) )                           

    ''' 입력준비 '''
    image = torch.from_numpy(image)
    mask  = torch.from_numpy(mask)

    pred = torch.from_numpy(np.zeros( image.shape[:-1]+(2,)))
    pred_count = torch.from_numpy(np.zeros_like(pred))

    mask_oh = torch.squeeze(mask,-1).to(torch.int64)
    mask_oh = torch.nn.functional.one_hot(mask_oh,num_classes=2).float()

    '''return'''
    return image, mask, mask_oh, pred, pred_count


        
def detransform_valid_output(image,mask,mask_oh,pred,pred_count,Dorg):

    ''' padding 제거'''
    image = image[:,:Dorg,:,:,:]
    mask = mask[:,:Dorg,:,:,:]
    pred = pred[:,:Dorg,:,:,:]
    pred_count = pred_count[:,:Dorg,:,:,:]
    mask_oh = mask_oh[:,:Dorg,:,:,:]

    ''' pred 평균 '''
    pred /= pred_count+1e-10

    ''' return '''
    return image, mask,mask_oh,pred