import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import cv2
import os
from skimage import io

MODEL_CHECKPOIN_PATH = 'checkpoints/cp-{epoch:04d}.ckpt'
MODEL_SAVEFILE_NAME = "trained_model.h5"

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

MODEL_CHECKPOINT_DIR = os.path.dirname(MODEL_CHECKPOIN_PATH)

res = {}
cal = 0

train = ImageDataGenerator(rescale=1./255)
validation = ImageDataGenerator(rescale=1./255)


def create_model_from_zero(steps, epochs):
    print("\n\t[MODEL] Create Model from Zero -> Start\n")

    train_dataset = train.flow_from_directory(
        "images", target_size=(300, 300), batch_size=5)

    validation_dataset = validation.flow_from_directory(
        "validation", target_size=(300, 300), batch_size=5)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            # tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(5, activation='sigmoid')
        ]
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_CHECKPOIN_PATH, save_weights_only=True, verbose=1, period=5)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print("\n\t\t- Started Fit Phase\n")
    model.fit(train_dataset, steps_per_epoch=steps, epochs=epochs,
              validation_data=validation_dataset, callbacks=[cp_callback])

    print("\n\t\t- Finished Fit Phase\n")
    model.save(MODEL_SAVEFILE_NAME)
    print("\n\t\t- Saved Model\n")
    print("\n\t[MODEL] Create Model from Zero -> End\n")

    return model


model = create_model_from_zero(30, 30)
print("[SUCCESS] MODEL IS READY")
