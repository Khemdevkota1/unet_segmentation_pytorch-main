import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from progressbar import ProgressBar
TRAIN_IMAGE_DATA_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/Training_images/"
TRAIN_MASK_DATA_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/Training_mask/"



NEW_TRAIN_IMAGE_DATA_PATH="/home/jeshanpokharel/Downloads/khimsir new data/train_images/"
NEW_TRAIN_MASK_DATA_PATH="/home/jeshanpokharel/Downloads/khimsir new data/train_masks/"



images_sorted=sorted(os.listdir(TRAIN_IMAGE_DATA_PATH))
mask_sorted=sorted(os.listdir(TRAIN_MASK_DATA_PATH))



#img=Image.open(TRAIN_IMAGE_DATA_PATH+images_sorted[0])
#plt.imshow(img)

#img=Image.open(TRAIN_MASK_DATA_PATH+mask_sorted[0])
#plt.imshow(img)
progress=ProgressBar(max_value=len(images_sorted))
for image,mask in zip(images_sorted,mask_sorted):
    image_img,mask_img=Image.open(TRAIN_IMAGE_DATA_PATH+image),Image.open(TRAIN_MASK_DATA_PATH+mask)
    mask_arr=np.array(mask_img)
    #print(bool(np.isin(85,mask_arr)))
    #if bool(np.isin(85,mask_arr))==True:
    #print(image,mask)
    mask_arr[mask_arr==255]=255
    mask_arr[mask_arr==85]=255
    mask_arr[mask_arr==170]=255
    #save mask
    img=Image.fromarray(mask_arr)
    img.save(NEW_TRAIN_MASK_DATA_PATH+f"{mask}")
    #save image
    image_img.save(NEW_TRAIN_IMAGE_DATA_PATH+f"{image}")
    progress.next()
        
        