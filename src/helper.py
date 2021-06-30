
from PIL import Image
import numpy as np, cv2, math, time
import os
    
import time

def imshow(image):
    try:
        display( Image.fromarray(np.uint8(np.squeeze(image))))
    except:
        pass
    
def imread(path):
    return cv2.imread( path, cv2.IMREAD_GRAYSCALE)

def setup_gpu(gpu_id:str):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    
def cooltime(key='default',cooltime_=1.0):
    now = time.time()
    if key in cooltime.prevs.keys():
        prev = cooltime.prevs[key]
    else:
        cooltime.prevs[key] = now
        prev = now
    
    if now>(prev+cooltime_):
        cooltime.prevs[key] = now
        return True
    else: 
        return False
cooltime.prevs = {}

# if __name__ == '__main__':
#     print(cool_time())
#     time.sleep(0.1)
#     print(cool_time())
#     time.sleep(2.0)
#     print(cool_time())


class StopWatch():
    def __init__(self):
        '''
        start stop reset get
        '''
        self.time_start = 0
        self.time_stop = 0
        self.run = 0
        return
    
    def start(self):
        self.time_start = time.time()
        return
    
    def stop(self):
        self.time_stop = time.time()
        self.run += self.time_stop - self.time_start
        return 
    
    def reset(self):
        self.time_start = 0
        self.time_stop = 0
        self.run = 0
        return 
    
    def get(self):
        return self.run
    

if __name__ == '__main__':
    watch = StopWatch()
    
    watch.start()
    time.sleep(1)
    watch.stop()
    print(watch.get())
    
    watch.start()
    time.sleep(1)
    watch.stop()
    print(watch.get())
        
    watch.start()
    time.sleep(1)
    watch.stop()
    print(watch.get())
    
    watch.reset()
    watch.start()
    time.sleep(1)
    watch.stop()
    print(watch.get())
