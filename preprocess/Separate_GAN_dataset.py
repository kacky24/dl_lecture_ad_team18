import numpy as np
import os
import shutil

rate_int = 36 # train : valid = a : 1‚Ì‚Æ‚«‚Éa+1‚ð“ü—Í
count = 0

path_snow =  'C:/DL2/data2/snow_japanese_talent/'
path_train = 'C:/DL2/data2/snow_japanese_talent_train/'
path_valid = 'C:/DL2/data2/snow_japanese_talent_valid/'

files = os.listdir('C:/DL2/data2/snow_japanese_talent')

path_original =       'C:/DL2/data2/original_japanese_talent/'
path_original_train = 'C:/DL2/data2/original_japanese_talent_train/'
path_original_valid = 'C:/DL2/data2/original_japanese_talent_valid/'

for file in files:
    count+=1
    
    # •\Ž¦—p
    if count%10==0:
        print(count)
        
    src_path = path_snow + file
    src2_path = path_original + file
    if count%rate_int != 0:
        dst_path = path_train + file
        shutil.copy2(src_path, dst_path)
        dst2_path = path_original_train + file
        shutil.copy2(src2_path, dst2_path)
    else:
        dst_path = path_valid + file
        shutil.copy2(src_path, dst_path)
        dst2_path = path_original_valid + file
        shutil.copy2(src2_path, dst2_path)  