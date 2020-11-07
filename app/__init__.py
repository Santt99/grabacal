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
MODEL_SAVEFILE_NAME = "trained_model.h5"

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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'PNG'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

res = {}
cal = 0

train = ImageDataGenerator(rescale=1./255)
train_dataset = train.flow_from_directory(
    "images", target_size=(300, 300), batch_size=32)


def create_model_from_save():
    print("\n\n\t[MODEL] Create Model from Save -> Start\n")
    model = keras.models.load_model(MODEL_SAVEFILE_NAME)
    print("\n\n\t[MODEL] Create Model from Save -> End\n")

    model.summary()

    # Re-evaluate the mol
    print("\n\n\t[MODEL] Reevaluate -> Start\n")
    # loss, acc = model.evaluate(train_dataset, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    print("\n\n\t[MODEL] End -> Start\n")
    return model


model = create_model_from_save()
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
    print("\n\t[MODEL] Train Indexes: ", train_dataset.class_indices, "\n")
    for i in os.listdir('test'):
        img = image.load_img('test/'+i, target_size=(300, 300))

        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        predictions = model.predict(images)
        loss, acc = model.evaluate(images)
        score = tf.nn.softmax(predictions[0])

        print("\n:: ================ STARTED PREDICTION =================== ::")
        print("\t- File: ", i)
        print("\t- Predictions: ", score)

        APPLE_PIE = 0
        FILET_MIGNON = 1
        FRENCH_FRIES = 2
        HAMBURGER = 3
        HOT_DOG = 4

        foodType = ''
        foodCalories = 0

        if(np.argmax(score) == APPLE_PIE):
            foodType = 'APPLE_PIE'
            foodCalories = 437

            valid = False
            for each in score.numpy():
                if each != np.float32(0.2):
                    valid = True

            if not valid:
                foodType = 'NO MATCH'
                foodCalories = 0
        elif(np.argmax(score) == FILET_MIGNON):
            foodType = 'FILET_MIGNON'
            foodCalories = 467
        elif(np.argmax(score) == FRENCH_FRIES):
            foodType = 'FRENCH_FRIES'
            foodCalories = 660
        elif(np.argmax(score) == HAMBURGER):
            foodType = 'HAMBURGER'
            foodCalories = 505
        elif(np.argmax(score) == HOT_DOG):
            foodType = 'HOT_DOG'
            foodCalories = 564

        print("\t- Class Predicted: ", foodType)
        print("")

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
        print("\n[MODEL] Starting to process")
        start_processing()
        print("[MODEL] Finished processing...\n")
        response = jsonify({"message": "Started processing images!"})
        return response, 200


def getActualCalories(image, calories):
    img = io.imread(image)[:, :, :-1]
    cols, rows = img.shape
    img = np.arange(cols % y)
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
