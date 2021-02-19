import os
import cv2 as cv
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam,SGD

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

path_train="images/train"
path_val="images/validation"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

params=[]
for i in os.listdir(path_train):
    params.append(i)
      
training_data=[]
for m in params:
        pat=os.path.join(path_val,m)
        for img in os.listdir(pat):
            img_path=os.path.join(pat,img)
            photo=load_img(img_path,target_size=(200,200))
            photo=img_to_array(photo)
            if m=='surprise':
               label=0
            elif m=='fear':
               label=1
            elif m=='angry':
                label=2
            elif m=='neutral':   
                label=3
            elif m=='sad':
                label=4
            elif m=='disgust':
                label=5
            elif m=='happy':
                label=6
            training_data.append([photo,label])
random.shuffle(training_data)
features=[]
labels=[]
for a,b in training_data:
    features.append(a)
    labels.append(b)
x_train=features[0:5000]
x_val=features[5000:7066]
y_train=labels[0:5000]
y_val=labels[5000:7066]
x_train=np.array(x_train)
x_val=np.array(x_val)
y_train=np.array(y_train)
y_val=np.array(y_val)
x_train=x_train/255
x_val=x_val/255

y_train = keras.utils.to_categorical(y_train, 7)
y_val=keras.utils.to_categorical(y_val,7)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(7, activation='softmax'))
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=80, verbose=2, validation_data=(x_val,y_val))
