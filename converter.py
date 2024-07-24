import cv2 as cv
import numpy as np
import tensorflow as tf

img=cv.imread('1.jpeg')
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img=cv.resize(img,(28,28))
img=cv.bitwise_not(img)
cv.imshow('win', img) 
cv.waitKey(0) 
cv.destroyAllWindows() 
arr=np.array(img)
arr=arr/255
arr = np.expand_dims(arr, axis=-1)
arr = np.expand_dims(arr, axis=0)

model=tf.keras.models.load_model("mnist_hand_Neural_model.h5")
def prediction(mat):
    logits=model(mat)
    f_x=tf.nn.softmax(logits)
    print(np.argmax(f_x))

prediction(arr)