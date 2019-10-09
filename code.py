#%%
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#%%

#%%
df=pd.read_csv("fashion-mnist_train.csv")

#%%
df.head()
#%%
image=df.iloc[0:1,1:]
#%%
image
#%%
import matplotlib.pyplot as plt                         #importing matplot for image fetching from pixel
plt.imshow(image.values.reshape(28,28))                 #since image are of 28*28 dimension(pixel)
#%%
train,test=train_test_split(df,test_size=0.2,random_state=12)
del df
