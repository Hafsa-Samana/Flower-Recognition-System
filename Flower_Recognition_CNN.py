#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
print(os.listdir('D:/flowers-recognition/flowers/Training_Testing'))


# In[3]:


X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='D:/flowers-recognition/flowers/Training_Testing/daisy'
FLOWER_SUNFLOWER_DIR='D:/flowers-recognition/flowers/Training_Testing/sunflower'
FLOWER_TULIP_DIR='D:/flowers-recognition/flowers/Training_Testing/tulip'
FLOWER_DANDI_DIR='D:/flowers-recognition/flowers/Training_Testing/dandelion'
FLOWER_ROSE_DIR='D:/flowers-recognition/flowers/Training_Testing/rose'


# In[4]:


def assign_label(img,flower_type):
    return flower_type


# In[5]:


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


# In[6]:


make_train_data('daisy',FLOWER_DAISY_DIR)
print(len(X))


# In[7]:


make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))


# In[8]:


make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))


# In[9]:


make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))


# In[14]:


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


# In[ ]:





# In[ ]:




