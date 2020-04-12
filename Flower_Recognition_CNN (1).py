#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
print(os.listdir('D:/flowers-recognition/flowers/Training_Testing'))


# In[4]:


X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='D:/flowers-recognition/flowers/Training_Testing/daisy'
FLOWER_SUNFLOWER_DIR='D:/flowers-recognition/flowers/Training_Testing/sunflower'
FLOWER_TULIP_DIR='D:/flowers-recognition/flowers/Training_Testing/tulip'
FLOWER_DANDI_DIR='D:/flowers-recognition/flowers/Training_Testing/dandelion'
FLOWER_ROSE_DIR='D:/flowers-recognition/flowers/Training_Testing/rose'


# In[5]:


def assign_label(img,flower_type):
    return flower_type


# In[6]:


import cv2
import numpy as np
def make_train_data(flower_type,DIR):
    for img in os.listdir(DIR):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))


# In[7]:


make_train_data('daisy',FLOWER_DAISY_DIR)
print(len(X))


# In[8]:


make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))


# In[9]:


make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))


# In[10]:


make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))


# In[12]:


import matplotlib.pyplot as plt
import random as rn
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()
plt.show()


# In[14]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255


# In[16]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[18]:


import tensorflow as tf
import random as rn
np.random.seed(42)
rn.seed(42)
tf.compat.v2.random.set_seed
tf.random.set_seed(42)


# In[22]:


# # modelling starts using a CNN.
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))


# In[24]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


# In[ ]:




