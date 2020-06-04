import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
####first
path = glob.glob("F:/deeplearning/brain_tumor_dataset/train/yes/*.JPG")
cv_img = []
for img in path:
    n = cv2.imread(img)
    #n=float(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(64,64))
    cv_img.append(b)
xtrain_set1= np.array(cv_img)
print(xtrain_set1.shape)
xtrain_flat1 = xtrain_set1.reshape(xtrain_set1.shape[0], -1).T
print(xtrain_flat1.shape)
train_set1 = xtrain_flat1.reshape(xtrain_flat1.shape[1], xtrain_flat1.shape[0],1)
print(train_set1.shape)
####second
path = glob.glob("F:/deeplearning/brain_tumor_dataset/train/no/*.JPG")
cv_img1 = []
for img in path:
    n = cv2.imread(img)
    #n=float(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(64,64))
    cv_img1.append(b)
xtrain_set2= np.array(cv_img1)
print(xtrain_set2.shape)
xtrain_flat2 = xtrain_set2.reshape(xtrain_set2.shape[0], -1).T
print(xtrain_flat2.shape)
train_set2 = xtrain_flat2.reshape(xtrain_flat2.shape[1], xtrain_flat2.shape[0],1)
print(train_set2.shape)
train_data= np.concatenate([train_set1,train_set2],axis=0)
print(train_data.shape)
###label
a1=[0]*xtrain_flat1.shape[1]
b1=[1]*xtrain_flat2.shape[1]
a2=np.array(a1)
b2=np.array(b1)
c=np.concatenate((a2,b2),axis=0)
c=c.reshape(-1,1)
train_label=c
print(train_label.shape)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64*64*3,1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_label, epochs=20)
