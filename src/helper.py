
from PIL import Image
import numpy as np, cv2, math
import os

def imshow(image):
    display( Image.fromarray(np.uint8(np.squeeze(image))))
def imread(path):
    return cv2.imread( path, cv2.IMREAD_GRAYSCALE)

def setup_gpu(gpu_id:str):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id