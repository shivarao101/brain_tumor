import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
trainy=0
trainn=0
path = glob.glob("F:/deeplearning/brain_tumor_dataset/train/yes/*.JPG")#Apple_Apple_scab
cv_img = []
for img in path:
    n = cv2.imread(img)
    #n=float(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(32,32))
    cv_img.append(b)
   
x_train_set1= np.array(cv_img)    
train_set_x_flat1 = x_train_set1.reshape(x_train_set1.shape[0], -1).T
print(train_set_x_flat1.shape)
path = glob.glob("F:/deeplearning/brain_tumor_dataset/train/no/*.JPG") #Apple_Black_rot
cv_img1 = []
for img in path:
    n = cv2.imread(img)
    #n=np.double(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(32,32))
    cv_img1.append(b)
  
x_train_set2= np.array(cv_img1)    
train_set_x_flat2 = x_train_set2.reshape(x_train_set2.shape[0], -1).T
print(train_set_x_flat2.shape)
train_data= np.concatenate([train_set_x_flat1,train_set_x_flat2],axis=1)
print(train_data.shape)
train_data1= train_data.reshape(train_data.shape[1],train_data.shape[0],1)
print(train_data1.shape)
a1=[0]*train_set_x_flat1.shape[1]
b1=[1]*train_set_x_flat2.shape[1]
a2=np.array(a1)
b2=np.array(b1)
c=np.concatenate((a2,b2),axis=0)
c=c.reshape(-1,1)
train_label=c
print(train_label.shape)
#train_label= np.concatenate([a1,b1],axis=1)
#print(train_label.shape)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3072,1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data1, train_label, epochs=20)

############testing
path = glob.glob("F:/deeplearning/brain_tumor_dataset/test/yes/*.JPG")#Apple_Apple_scab
cv_img = []
for img in path:
    n = cv2.imread(img)
    #n=float(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(32,32))
    cv_img.append(b)
   
x_test_set1= np.array(cv_img)    
test_set_x_flat1 = x_test_set1.reshape(x_test_set1.shape[0], -1).T
print(test_set_x_flat1.shape)
path = glob.glob("F:/deeplearning/brain_tumor_dataset/test/no/*.JPG") #Apple_Black_rot
cv_img1 = []
for img in path:
    n = cv2.imread(img)
    #n=np.double(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(32,32))
    cv_img1.append(b)
  
x_test_set2= np.array(cv_img1)    
test_set_x_flat2 = x_test_set2.reshape(x_test_set2.shape[0], -1).T
print(test_set_x_flat2.shape)
test_data= np.concatenate([test_set_x_flat1,test_set_x_flat2],axis=1)
print(test_data.shape)
test_data1= test_data.reshape(test_data.shape[1],test_data.shape[0],1)
print(test_data1.shape)
a1=[0]*test_set_x_flat1.shape[1]
b1=[1]*test_set_x_flat2.shape[1]
a2=np.array(a1)
b2=np.array(b1)
c=np.concatenate((a2,b2),axis=0)
c=c.reshape(-1,1)
test_label=c
test_loss, test_acc=model.evaluate(test_data1,  test_label, verbose=2)
print('\nTest accuracy:', test_acc, test_loss)
