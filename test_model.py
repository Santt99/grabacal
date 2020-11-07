import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# https://tfhub.dev/google/aiy/vision/classifier/food_V1/1
# https://tfhub.dev/google/experts/bit/r50x1/in21k/food/1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

m = tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')
])
m.build([None, 192, 192, 3])
for i in os.listdir('test'):
    img = image.load_img('test/'+i, target_size=(192, 192))
    # plt.imshow(img)
    # plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    predictions = m.predict(images)
    classe = m.predict_classes(images)
    print("CLASS: ", classe)
    print("PREDICTIONS (", i, "): ", np.argmax(predictions, axis=1))
