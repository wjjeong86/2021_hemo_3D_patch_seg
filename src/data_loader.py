


'''
data loader 만들기

https://wikidocs.net/57165

커스텀 데이터셋
torch.utils.data.Dataset 상속받아 만듦.
아래 기본 포맷
class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self):   데이터 전처리
  def __len__(self):    총 샘플의 수를 리턴
  def __getitem__(self, idx):    샘플 1개 가져오는 함수
  
torch 데이터는 보통..
    NCDHW
    NCHW
나는 익숙하지가 않네.그래서 아는대로 조작해야겠다.
    NDHWC
    NHWC
    

Dataset과 DataLoader. 
Dataset은 map 스타일과 iter 스타일이 있는데.
map 은 배열이라고 보면 되고.
iter 는 generator라고 보면 된다.
훈련시는 iter 쓰면 편할거고 보통은 map 쓰면 될거 같네.
DataLoader는 안에 sampler, collec_fn 등등을 넣어줄수 있는것 같고.
sampler는 샘플 고르는 순서 를 정해주나보다. 랜덤을 여기서 구현하면 되는데.. 굳이??
Dataset과 DataLoader 사이에는 간극이 있다. Dataset은 샘플 1개를 만드는거고
DataLoader는 batch(:샘플여러게 모아서) 를 만든다. collect_fn이 여러 샘플을 배치로 합쳐주나 봄.


DataLoader끼리 worker를 공유하는게 아닐까 싶음. 
DataLoader <- DataLoader 로 직렬 연결된 구조에서  둘다 worker>=1 병렬 처리 옵션을 켜줄때 오류가 발생한다. 키값 에러라고.
해결책.. Dataset 수준에서는 직렬로 연결하고 맨 마지막에 DataLoader를 붙여준다.

(원래계획)
환자CT Dataset -> DataLoader -> 패치 Dataset -> DataLoader
(수정계획)
환자CT Dataset -> 패치 Dataset -> DataLoader

DataLoader를 병렬로 연결하는 것은 허용하는 모양이다.
     
+

  
  
  
'''

import pandas as pd, numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from helper import *

        
''' =========================================================== dataset class '''
def fn_setup_seed(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
        
class HemoPatientDatasetBase():
    ''' 한명한명 
    @ meta :
    @ image_size :
    @ copy_samples :
    '''
    
    
    '''-------------------------구현필수'''
    def __init__(self, meta, image_size, copy_samples:int):
        super().__init__()
        self.meta = meta
        self.isize = image_size
        self.copy_samples = copy_samples
        self.list_id_patient = np.unique(meta.id_patient)
        return
    
    def __len__(self):
        return len(self.list_id_patient)
    
    def __getitem__(self,idx):
        id_patient = self.list_id_patient[idx]
        while True:
            try:
                item = self.get_a_patient(id_patient)
            except:
                print('로딩실패:',id_patient)
            else:
                break
        return item

    def __iter__(self):
        while True:
            r = np.random.randint(0,len(self.list_id_patient))
            id_patient = self.list_id_patient[r]
            item = self.get_a_patient(id_patient)
            for i_copy in range(self.copy_samples):
                yield item
        return
            
        
    
    
    '''-------------------------'''
    def get_a_patient(self,id_patient):
        worker_info = torch.utils.data.get_worker_info()
        
        ##### 1명 환자 정리
        meta_a_patient = self.meta[ self.meta.id_patient==id_patient]
        meta_a_patient = meta_a_patient.sort_values(by='axis_z', ascending=False)
        
        ##### patient 전체에 대한 정보
        is_hemo_patient = meta_a_patient.is_hemo_patient.iloc[0] #문제없다면 0번만봐도 된다.
        
        ##### slice 각각
        images, masks, paths_image = [], [], []
        
        for _, a_slice in meta_a_patient.iterrows():
            
            path_image = a_slice.path_ct
            image = imread( path_image)
            if a_slice.is_hemo_slice:
                mask = imread(a_slice.path_mask)
            else:
                mask = np.zeros_like(image)
            
            image = cv2.resize(image, self.isize)
#             mask = cv2.dilate(mask,np.ones((7,7)))
            mask = cv2.resize(mask, self.isize, cv2.INTER_NEAREST)
            
            ##### append
            images.append( np.expand_dims(image,-1) )
            masks.append( np.expand_dims(mask,-1) )
            paths_image.append( path_image)
            
        
        ##### concat 
        images = np.float32(np.stack(images,0))/255.0
        masks  = np.float32(np.stack(masks ,0))/255.0
        
#         D,H,W,C = images.shape 
#         images = np.reshape(images, (1,D,H,W,C) )
#         masks  = np.reshape(masks , (1,D,H,W,C) )
        
        #####
        return is_hemo_patient, images, masks, list(meta_a_patient.index)
    
    
        
        
class HemoPatientDataset(HemoPatientDatasetBase, Dataset):
    def __init__(self, meta, image_size, copy_samples:int):
        Dataset.__init__(self)
        HemoPatientDatasetBase.__init__(self,meta,image_size,copy_samples)
        return
    
    
        
class HemoPatientIterableDataset(HemoPatientDatasetBase, IterableDataset):
    def __init__(self, meta, image_size, copy_samples:int):
        IterableDataset.__init__(self)
        HemoPatientDatasetBase.__init__(self,meta,image_size,copy_samples)
        return
        
        
    

        

    
''' ======================================= 랜덤패치 추출하는  '''
class HemoRandomPatchIterableDataset(IterableDataset):
    def __init__(self,iter_patient,sample_from_patient,patch_size, mode_mask_weight=False):
        ''' 
        환자 1명 데이터로부터 여러개의 랜덤 패치 추출.
        효율상 여러개의 랜덤 패치를 얻어야 된다. 그래서
        항상 배치를 꽉 채워서 배출..
        뒤에 오는 DataLoader는 batch_size=None로 설정해야 함
        
        @
        sample_from_patient : 한 환자로부터 몇개 샘플을 뽑을건지
        
        mode_hemo_weight : 
          True : 절반은 마스크 있는 부위에 가중치, 절반은 신경 안쓰고 다 뽑음
          False : 마스크 신경 안쓰고 다 뽑음
          
        '''
        super().__init__()
        self.iter_patient = iter_patient
        self.mode_mask_weight = mode_mask_weight
        self.sample_from_patient = sample_from_patient
        self.patch_size = patch_size
    
#     def __len__(self):
#         return self.patches_per_patient
    
#     def __getitme__(self,idx):

    def __iter__(self):        
        while True:
            item =  self.get_random_patch()
            for i_patch in range(self.sample_from_patient):
                yield [i[i_patch] for i in item]
        return
        
        
    def get_random_patch(self):
        worker_info = torch.utils.data.get_worker_info()
        
        ##### get random patinet datat
        is_hemo_patient, arr4_image, arr4_mask, paths_ct = next(self.iter_patient)   
        arr4_image = np.array(arr4_image)
        arr4_mask = np.array(arr4_mask)
            
        ##### get patch : mask weight에 반응하는 부분
        if self.mode_mask_weight == True: weight = arr4_mask.reshape(-1)/arr4_mask.sum()
        else:             weight = np.reshape(np.ones_like(arr4_mask)/arr4_mask.size,-1)
            
        arr5_pimage, arr5_pmask = self.crop_random_patch( 
            arr4_image, arr4_mask, weight, sample_from_patient=self.sample_from_patient,
            psize = self.patch_size, 
        )
        
        return arr5_pimage, arr5_pmask
    
    
    
    def crop_random_patch(self, arr4_image, arr4_mask, weight, sample_from_patient, psize ):
        pd,ph,pw = psize
        md,mh,mw,mc = arr4_mask.shape     
        
        ##### 패딩
        arr4_image_pad = np.pad(arr4_image, [[pd//2,pd//2],[ph//2,ph//2],[pw//2,pw//2],[0,0]])
        arr4_mask_pad  = np.pad( arr4_mask, [[pd//2,pd//2],[ph//2,ph//2],[pw//2,pw//2],[0,0]])
        
        ##### 루프 시작
        patchs_image,patchs_mask = [],[]
        R = np.random.choice( list(range(arr4_mask.size)), size=sample_from_patient, p=weight)
        
        for r in R:
            
            ##### 랜덤 pick
            w = r%mw;       r = r//mw;
            h = r%mh;       r = r//mh;
            d = r
            
            ##### crop            
            sub_image = arr4_image_pad[ d:d+pd, h:h+ph, w:w+pw, :]
            sub_mask  = arr4_mask_pad [ d:d+pd, h:h+ph, w:w+pw, :]
            
            ##### update
            patchs_image.append( sub_image)
            patchs_mask.append(  sub_mask)
               
        ##### stack
        arr5_pimage = np.stack(patchs_image,0)
        arr5_pmask  = np.stack(patchs_mask,0)
        
        return arr5_pimage, arr5_pmask
    
    
    
    
''' ======================================= 훈련용 데이터셋  '''
class HemoRandomPatchTrainIterableDataset(IterableDataset):
    def __init__(self,iter_a,iter_b,iter_c):
        ''' 
        3가지 종류 데이터를 섞어서 배출
            - hemo, mask에 가중치
            - hemo, 균등가중치
            - normal 균등 가중치
        @
          
        '''
        super().__init__()
        self.iter_a = iter_a
        self.iter_b = iter_b
        self.iter_c = iter_c
    
#     def __len__(self):
#         return self.patches_per_patient
    
#     def __getitme__(self,idx):

    def __iter__(self):  
        while True:
            yield next(self.iter_a)
            yield next(self.iter_b)
            yield next(self.iter_c)
        return

            
     
    
''' ============================================================= DataLoader + Queue 가능한지 확인'''
if __name__ == '__main__':
    
    '''============================================================= meta csv 준비 '''
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
            
    '''========================================================= ds patient 확인'''
    ds_hemo_patient = HemoPatientDataset(
        meta_hemo, image_size=(128,128), copy_samples=1
    )
    ds_normal_patient = HemoPatientDataset(
        meta_normal, image_size=(128,128), copy_samples=1
    )
       
    ### 같은 환자가 copy_sample 만큼 나와야 된다.
    for i,(_,himage4,_,idx_meta) in enumerate(iter(ds_hemo_patient)):
        imshow( np.hstack(himage4)*255)        
        if i>=10: break;
            
    dl_hemo_patient = DataLoader( 
        ds_hemo_patient, batch_size=1, num_workers=1,
    )
    
    for i,(_,_,_,idx_meta) in enumerate(dl_hemo_patient):
        if i>=10: break;
            
    
    
    
    '''========================================================= ds patch 확인'''
    ds_hemo_patch = HemoRandomPatchIterableDataset(
        iter(ds_hemo_patient), mode_mask_weight=True, sample_from_patient=8,
    )
    ds_normal_patch = HemoRandomPatchIterableDataset(
        iter(ds_normal_patient), mode_mask_weight=False, sample_from_patient=8,
    )
    
#     for i,(image,mask) in enumerate(iter(ds_hemo_patch)):
#         S = np.hstack(image)
#         imshow(S*255)
#         if i>=18: break;


    '''========================================================= dl확인 :ds->ds->dl 구조'''

    
    dl_hemo_patch = DataLoader( 
        ds_hemo_patch, batch_size=8, num_workers=4,
        worker_init_fn=fn_setup_seed,
    )
    dl_normal_patch = DataLoader(
        ds_normal_patch, batch_size=8, num_workers=4,
        worker_init_fn=fn_setup_seed,
    )
    
    # iter를 꺼내놔야지 빠르다.
    iter_hemo = iter(dl_hemo_patch)
    iter_normal = iter(dl_normal_patch)
    
    for i,((himage5,hmask5),(nimage5,nmask5)) in \
        enumerate( zip(iter_hemo,iter_normal) ):
        
        himage = himage5[0,3,:]
        hmask = hmask5[0,3,:]
        nimage = nimage5[0,3,:]
        nmask = nmask5[0,3,:]
        
        summry = np.hstack( (himage,hmask,nimage,nmask))
        imshow(summry*255)
         
        if i>=30:
            break;