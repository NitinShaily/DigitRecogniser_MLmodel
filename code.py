#%%
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#%%

#%%
df=pd.read_csv("fashion-mnist_train.csv")


                                 
#%%
image                                                   #see array of pixel values of digit at first row


#%%
train,test=train_test_split(df,test_size=0.2,random_state=12)
del df

#%%
def x_and_y(df):
    x=df.drop(['label'],axis=1)      #dropping of target columns
    y=df.label 
    return x,y
x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)


#%%
#training of our dataset and caluculation of accuracy

model=RandomForestClassifier(n_estimators=100,random_state=12)              
model.fit(x_train,y_train)
prediction=model.predict(x_test)
score=accuracy_score(y_test,prediction)
print(score)

#%%
#data visualisation

image=traina.iloc[0:1,1:]                                   #selecting 1st row except colm 1 which is of label (target variable)

#%%
import matplotlib.pyplot as plt
plt.imshow(image.values.reshape(28,28))                     #cuz mnist dataset have image of 28*28 pixel
#%%
prediction=model.predict(image)
print(prediction)
