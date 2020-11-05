import os
from flask import Flask, flash, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './test'
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
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

MODEL_CHECKPOINT_DIR = os.path.dirname(MODEL_CHECKPOIN_PATH)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'PNG'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

res = {}
cal = 0

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    "training", target_size=(500, 500), batch_size=32, class_mode='sparse')
validation_dataset = validation.flow_from_directory(
    "validation", target_size=(500, 500), batch_size=32, class_mode="sparse")
print("[MODEL] Indexes: ", validation_dataset.class_indices)


def create_model_from_zero(steps, epochs):
    print("[MODEL] Create Model from Zero -> Start")
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', input_shape=(500, 500, 3)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                128, activation='relu'),
            tf.keras.layers.Dense(
                1, activation='sigmoid')
        ]
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_CHECKPOIN_PATH, save_weights_only=True, verbose=1, period=5)

    print("\t- Started Compile Phase")
    model.compile(loss='sparse',
                  optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
    print("\t- Finished Compile Phase")
    print("\t- Started Fit Phase")
    model.fit(train_dataset, steps_per_epoch=steps, epochs=epochs,
              validation_data=train_dataset, callbacks=[cp_callback])
    print("\t- Finished Fit Phase")
    print("\t- Started Model Save Phase")
    model.save(MODEL_SAVEFILE_NAME)
    print("\t- Finished Model Save Phase")
    print("[MODEL] Create Model from Zero -> End")
    img = image.load_img('test.jpg', target_size=(500, 500))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    loss, acc = model.evaluate(images)
    print("[MODEL] accuracy: {:5.2f}%".format(100*acc))

    return model


def create_model_from_checkpoint():
    latest = tf.train.latest_checkpoint(MODEL_CHECKPOINT_DIR)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', input_shape=(500, 500, 3)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                128, activation='relu'),
            tf.keras.layers.Dense(
                1, activation='sigmoid')
        ]
    )

    model.load_weights(latest)
    return model


def create_model_from_save():
    print("[MODEL] Create Model from Save -> Start")
    model = keras.models.load_model(MODEL_SAVEFILE_NAME)
    print("[MODEL] Create Model from Save -> End")
    img = image.load_img('test.jpg', target_size=(500, 500))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    loss, acc = model.evaluate(images)
    print("[MODEL] accuracy: {:5.2f}%".format(100*acc))

    return model


model = create_model_from_zero(10, 30)
#model = create_model_from_checkpoint()
# model = create_model_from_save()
print("[SUCCESS] MODEL IS READY")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            response = jsonify(
                'No image provided with "image" key, please use the correct key')
            return response, 400
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            response = jsonify('No selected image, please provide one')
            return response, 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            response = jsonify({"imageName": filename})
            return response, 200
        else:
            response = jsonify("That file format is not allowed")
            return response, 400


def start_processing():
    for i in os.listdir('test'):
        img = image.load_img('test/'+i, target_size=(500, 500))

        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        val = model.predict(images)
        loss, acc = model.evaluate(images)

        APPLE_PIE = 0
        FILET_MIGNON = 1
        FRENCH_FRIES = 2
        HAMBURGER = 3
        HOT_DOG = 4

        foodType = ''
        foodCalories = 0

        if(val == APPLE_PIE):
            foodType = 'APPLE_PIE'
            foodCalories = 437
        elif(val == FILET_MIGNON):
            foodType = 'FILET_MIGNON'
            foodCalories = 467
        elif(val == FRENCH_FRIES):
            foodType = 'FRENCH_FRIES'
            foodCalories = 660
        elif(val == HAMBURGER):
            foodType = 'HAMBURGER'
            foodCalories = 505
        elif(val == HOT_DOG):
            foodType = 'HOT_DOG'
            foodCalories = 564

        foodCalories = getActualCalories('test/'+i, foodCalories)

        res[i] = {
            "food_type": foodType,
            "model_accuracy": acc,
            "food_calories_aprox": foodCalories
        }

    with open('res.json', 'w') as json_file:
        json.dump(res, json_file)


@app.route('/results', methods=['GET'])
def get_results():
    if request.method == 'GET':
        with open('res.json') as json_file:
            data = json.load(json_file)
            if not bool(data):
                response = jsonify({"message": "Still processing!"})
                return response, 200
            else:
                response = jsonify({"message": "Done!", "data": data})
                return response, 200


@app.route('/process', methods=['GET'])
def process():
    if request.method == 'GET':
        start_processing()
        response = jsonify({"message": "Started processing images!"})
        return response, 200


def getActualCalories(image, calories):
    img = io.imread(image)[:, :, :-1]
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    redValue = dominant[0]
    greenValue = dominant[1]

    if redValue > greenValue:
        return calories + 200
    elif greenValue > redValue:
        return calories - 200
    else:
        return calories + 100
