import pandas as pd, os

from torch.utils.data import DataLoader
from data_loader import HemoPatientDataset, HemoPatientIterableDataset, HemoRandomPatchIterableDataset, HemoRandomPatchTrainIterableDataset, fn_setup_seed

# def get_meta_SNU():
#     ##### meta csv 준비
#     path_hemo_csv = '/mnt/ssd0/Hemorrhage/Data/00_SNU/SN_Hemo_Data.csv'
#     path_normal_csv = '/mnt/ssd0/Hemorrhage/Data/00_SNU/SN_Normal_Data_v2.csv'
#     meta_hemo = pd.read_csv( path_hemo_csv, index_col=None)
#     meta_normal = pd.read_csv( path_normal_csv, index_col=None)
#     meta = pd.concat( [meta_hemo,meta_normal] )
#     meta = meta.reset_index(drop=True)

#     ##### 경로 오류 수정
#     for i in meta.index:
#         meta.at[i,'path_ct'] = meta.at[i,'path_ct'].replace('mnt_ssd1','ssd0')
#         if str == type(meta.at[i,'path_mask']):
#             meta.at[i,'path_mask'] = meta.at[i,'path_mask'].replace('mnt_ssd1','ssd0')
#         else:
#             meta.at[i,'path_mask'] = 'no_mask'
# #     meta_hemo = meta[meta['is_hemo_patient']==True].reset_index(drop=True)
# #     meta_normal = meta[meta['is_hemo_patient']==False].reset_index(drop=True)
    
#     return meta


def get_meta(path_csv):
    
    dir_ = os.path.split(path_csv)[0]

    ##### meta 파일 로드 
    meta = pd.read_csv(path_csv)
    meta = meta[ (meta['src_data']=='AJ') | (meta['src_data']=='SN') ]
    meta = meta.reset_index(drop=True)
    
    ##### 경로 수정
    for i in meta.index:
        
        meta.at[i,'path_ct'] = os.path.join(dir_,meta.at[i,'path_ct'][1:])
        
        if str == type(meta.at[i,'path_mask']):
            meta.at[i,'path_mask'] = os.path.join(dir_,meta.at[i,'path_mask'][1:])
        else:
            meta.at[i,'path_mask'] = 'no_mask'
                        
    
    ##### meta 파일 
    meta_hemo = meta[meta['is_hemo_patient']==1]
    meta_normal = meta[meta['is_hemo_patient']!=1]

    ##### train valid
    meta_trh = meta_hemo[meta_hemo['n_fold']<8]
    meta_vlh = meta_hemo[meta_hemo['n_fold']==8]
    meta_teh = meta_hemo[meta_hemo['n_fold']==9]

    meta_trn = meta_normal[meta_normal['n_fold']<8]
    meta_vln = meta_normal[meta_normal['n_fold']==8]
    meta_ten = meta_normal[meta_normal['n_fold']==9]
    
    return meta_trh, meta_vlh, meta_teh, meta_trn, meta_vln, meta_ten


def get_data_loader_train(meta_trh, meta_trn, BATCH_SIZE, IMAGE_SIZE, N_PATCH_PER_PATIENT, PATCH_SIZE):
    
    '''dataloader train'''
    ##### ds->ds->dl 구조'''
    ds_trh_patient = HemoPatientIterableDataset(meta_trh, IMAGE_SIZE, 1)
    ds_trn_patient = HemoPatientIterableDataset(meta_trn, IMAGE_SIZE, 1)

    ds_trh_patch_1 = HemoRandomPatchIterableDataset( iter(ds_trh_patient), N_PATCH_PER_PATIENT, PATCH_SIZE, mode_mask_weight=True)
    ds_trh_patch_2 = HemoRandomPatchIterableDataset( iter(ds_trh_patient), N_PATCH_PER_PATIENT, PATCH_SIZE, mode_mask_weight=False)
    ds_trn_patch   = HemoRandomPatchIterableDataset( iter(ds_trn_patient), N_PATCH_PER_PATIENT, PATCH_SIZE, mode_mask_weight=False)

    ds_tr = HemoRandomPatchTrainIterableDataset( iter(ds_trh_patch_1), iter(ds_trh_patch_2), iter(ds_trn_patch))

    dl_tr = DataLoader( ds_tr, batch_size=BATCH_SIZE, num_workers=1, worker_init_fn=fn_setup_seed)

    return dl_tr
    
    