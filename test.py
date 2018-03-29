#######################################################################################
# @Purpose:  Predict the Characters in The Simposons 
# @Author:   Tai Rong <t106368030@ntut.edu.tw>
# @Date:     2017/10/25
# @package:  Keras 2.0 , Opencv 3.0 ,numpy 1.13.3
# @Input:    test images:  test.zip >> test (10.6 MB) 
#                           /
#                           |_test.py
#                           |_test/
#                              |_test/
#                                 |_1.jpg ~ 999.jpg
#            test Label:   listdir.txt
########################################################################################
import cv2
import os
import numpy as np
from keras.models import load_model
import pandas as pd


def read_images(path):
    images=[]
    for i in range(990):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
        images.append(image)

    images=np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range (lenSIZE):
        temp=listdir[label[i]]
        label_str.append(temp)

    return label_str

images = read_images('test/test/')
model = load_model('h5/50_debug256debug.h5')

predict = model.predict_classes(images, verbose=1)
print(predict)
label_str=transform(np.loadtxt('listdir.txt',dtype='str'),predict,images.shape[0])

pd.DataFrame({"id": list(range(1,len(label_str)+1)),"character": label_str}).to_csv('test_score.csv', index=False, header=True)
