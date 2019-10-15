#using matplotlib:
image=traina.iloc[0:1,1:]              #select the first row of mnist data and put it in image

image

import matplotlib.pyplot as plt               
plt.imshow(image.values.reshape(28,28)) #print the image using pixels/data given in size of 28*28 cuz row contain 784(=28*28)

prediction=model.predict(image)       #prediction of given image
print(prediction)

import matplotlib.image as mpimg

i=mpimg.imread("ima.png")                #reading a image using matplot

plt.imshow(i)                            
