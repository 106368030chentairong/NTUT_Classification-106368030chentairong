#######################################################################################
# @Purpose:  Predict the Characters in The Simposons 
# @Author:   Tai Rong <t106368030@ntut.edu.tw>
# @Date:     2017/10/25
# @package:  Keras 2.0 , TensorFlow 1.3 , Opencv 3.0 , sklearn 0.18 ,numpy 1.13.3
# @Input:    Train images:  train.zip >> characters-20 (536 MB) 
#                           /
#                           |_main.py
#                           |_train/
#                              |_characters-20/
#                                 |_abraham_grampa_simpson/XXX.jpg ...
#            Train Label:   listdir.txt
########################################################################################
import os,sys
import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model, load_model
from keras import applications
from keras.layers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

epochs=100
batch_size=256

images=[]
labels= []
listdir= []

def read_images_labels(path,i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))   # abs_path =  C:\\XXX\XXX\ + train\XXX\  ||  +(XXX).jpg 
        if os.path.isdir(abs_path):
            i+=1                                               # 1- 20
            temp = os.path.split(abs_path)[-1]                 # C:\\XXX\XXX\ + train\XXX\ >> XXX
            listdir.append(temp)                               # stack file path
            read_images_labels(abs_path,i)                     # read_images_labels(C:\\XXX\XXX\ + train\XXX\)
            amount = int(len(os.listdir(path)))              # train\ file amount
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp) #Loading Bar
        else:  
            if file.endswith('.jpg'):
                image=cv2.resize(cv2.imread(abs_path),(64,64)) # read XXX.jpg resize 64x64
                images.append(image)                           # stack image
                labels.append(i-1)                             # stack labels
    return images, labels ,listdir

def read_main(path):
    images, labels ,listdir = read_images_labels(path,i=0)
    images = np.array(images,dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=20)
    np.savetxt('listdir.txt', listdir, delimiter = ' ',fmt="%s")
    return images, labels

images, labels=read_main('train/characters-20')
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(20, activation='sigmoid'))
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True,)
datagen.fit(X_train)

file_name=str(epochs)+'_debug'+str(batch_size)
TB=TensorBoard(log_dir='logs/'+file_name)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=epochs, epochs=epochs,
                    validation_data = (X_test, y_test ), verbose = 1,callbacks=[TB])
model.save('h5/'+file_name+'debug.h5')
score = model.evaluate(X_test, y_test, verbose=0)
print(score)