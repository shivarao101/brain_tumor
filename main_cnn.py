import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
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

train_data= np.concatenate(((cv_img),(cv_img1)),axis=0)
print(train_data.shape)
###label
a1=[0]*len(cv_img)
b1=[1]*len(cv_img1)
a2=np.array(a1)
b2=np.array(b1)
c=np.concatenate((a2,b2),axis=0)
c=c.reshape(-1,1)
train_label=c
print(c)
print(train_label.shape)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.fit(train_data, train_label, epochs=20)

#####testing
path = glob.glob("F:/deeplearning/brain_tumor_dataset/test/yes/*.JPG")
cv_img = []
for img in path:
    n = cv2.imread(img)
    #n=float(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(64,64))
    cv_img.append(b)
xtrain_set1= np.array(cv_img)

####second
path = glob.glob("F:/deeplearning/brain_tumor_dataset/test/no/*.JPG")
cv_img1 = []
for img in path:
    n = cv2.imread(img)
    #n=float(n)/255
    #a= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    a=np.double(n)/255
    b=cv2.resize(a,(64,64))
    cv_img1.append(b)

test_data= np.concatenate(((cv_img),(cv_img1)),axis=0)

a1=[0]*len(cv_img)
b1=[1]*len(cv_img1)
a2=np.array(a1)
b2=np.array(b1)
c=np.concatenate((a2,b2),axis=0)
c=c.reshape(-1,1)
test_label=c
test_loss, test_acc=model.evaluate(test_data,  test_label, verbose=2)
print('\nTest accuracy:', test_acc, test_loss)

