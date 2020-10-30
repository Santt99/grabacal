from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 
import os

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset=train.flow_from_directory("training", target_size = (500,500),batch_size = 10,class_mode = 'binary')  
validation_dataset=validation.flow_from_directory("validation", target_size = (500,500),batch_size = 10,class_mode = "binary")  
print(validation_dataset.class_indices)
model= tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (500,500,3)), 
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512,activation='relu'),
tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss = 'binary_crossentropy',optimizer= RMSprop(lr=0.001),metrics=['accuracy'])
model_fit = model.fit(train_dataset,steps_per_epoch = 3,epochs = 10,validation_data = train_dataset)


for i in os.listdir('test'):
    img = image.load_img('test/'+i,target_size=(500,500))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X,axis = 0)
    images = np.vstack([X]) 
    val = model.predict(images)
    if (val==1):
        print("pizza")
    else:
        print("pie")