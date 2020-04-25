#!/usr/bin/env python
# coding: utf-8

# # FLOWER RECOGNITION SYSTEM USING CNN

# # UPLOAD DATASET

# In[24]:


import os
print(os.listdir('D:/flowers-recognition/flowers/DATA'))


# # UPLOADING EACH FLOWER

# In[25]:


X=[]   # array for getting images 
Z=[]   # array for getting lable
IMG_SIZE=150
daisy='D:/flowers-recognition/flowers/DATA/daisy'
sunflower='D:/flowers-recognition/flowers/DATA/sunflower'
tulip='D:/flowers-recognition/flowers/DATA/tulip'
dandelion='D:/flowers-recognition/flowers/DATA/dandelion'
rose='D:/flowers-recognition/flowers/DATA/rose'


# # TYPE OF FLOWER

# In[26]:


def assign_label(img,flower_type):
    return flower_type


# # PRE-PROCESSING

# In[27]:


import cv2
import numpy as np
def train_data(flower_type,DIR):
    for img in os.listdir(DIR):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)  # taking images from the directory defined
        img = cv2.imread(path,cv2.IMREAD_COLOR) # convert tha rgb image into grayscale
        #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  # resizing the images
        
        X.append(np.array(img))
        Z.append(str(label))


# # TRAINING OF DATA

# In[28]:


train_data('daisy',daisy)
print(len(X))


# In[29]:


train_data('Sunflower',sunflower)
print(len(X))


# In[30]:


train_data('Tulip',tulip)
print(len(X))


# In[31]:


train_data('dandelion',dandelion)
print(len(X))


# In[32]:


train_data('Rose',rose)
print(len(X))


# In[33]:


import matplotlib.pyplot as plt
import random as rn
fig,ax=plt.subplots(5,2) # plotting each flower
fig.set_size_inches(15,15) # figure height and width is set though image is 150
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z)) # it will generate flower labels randomly
        ax[i,j].imshow(X[l])   # getting images
        ax[i,j].set_title('Flower: '+Z[l])  # getting images and assigning labels
        
plt.tight_layout()
plt.show()   # plotting the images


# In[34]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255


# # TRAINING AND TESTING

# In[35]:


from sklearn.model_selection import train_test_split   # we are not dividing our datset manually 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# # LENGTH OF TRAINING DATA

# In[45]:


print(len(x_train),(y_train))


# # LENGTH OF TESTING DATA

# In[44]:


print(len(x_test),(y_test))


# # SHAPE OF DATA

# In[41]:


print("Train data shape: {}".format(x_train.shape))
print("testing/validation data shape: {}".format(x_test.shape))


# # CONVOLUTIONAL NEURAL NETWORK

# In[40]:


from keras.models import Sequential
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense
input_shape = (150,150,3) #image width and hieght and 3rd is about filters
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2))) # reduce the number of parameters

model.add(Conv2D(filters = 64,kernel_size = (3,3), activation='relu')) # relu to convert the negative values as non-negative
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2))) #Stride is the number of pixels shifts over the input matrix.
model.add(Dropout(0.4))

model.add(Conv2D(filters = 96 , kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters = 96 , kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, (1,1), activation='relu'))
#model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax")) #apply Softmax function to classify an object with probabilistic values between 0 and 1(like a flower image or not)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# # ACCURACY

# In[81]:


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size = 64) # (number of times we are going to scan our whole training data)


# Our Training Accuracy comes out to be 98.27% and our Validation Accuracy comes out to be 70.21% which is good.

# # PREDICTION

# In[82]:


prediction=model.predict(x_test)
prediction_digits=np.argmax(pred,axis=1)


# # FLOWER CLASSES

# In[92]:


i=0   # starts from 0
proper_class=[]    # array for maintaining the trained fower
misclassified_class=[] # array for maintaining the testing flower

for i in range(len(y_test)):    
    if(np.argmax(y_test[i])==prediction_digits[i]):
        proper_class.append(i)
    if(len(proper_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==prediction_digits[i]):
        misclasified_class.append(i)
    if(len(misclassified_class)==8):
        break


# # CLASSIFICATION

# In[93]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[proper_class[count]]) #it will count the each flower
        ax[i,j].set_title("Predicted Flower :"+str(le.inverse_transform([prediction_digits[proper_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform([np.argmax(y_test[proper_class[count]])])))
        plt.tight_layout()
        count+=1


# So, the flower images are recognised correctly
