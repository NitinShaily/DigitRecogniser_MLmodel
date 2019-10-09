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
train,test=train_test_split(df,test_size=0.2,random_state=12)
del df


#%%
def x_and_y(df):
    x=df.drop(['label'],axis=1)      #dropping of unnecessary columns
    y=df.label 
    return x,y
x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)

#%%
y_train.head()

#%%
x_test.head()

#%%
model=RandomForestClassifier(n_estimators=100,random_state=12)              #main implementation of decision tree in our trained data
model.fit(x_train,y_train)
prediction=model.predict(x_test)
score=accuracy_score(y_test,prediction)
print(score)

