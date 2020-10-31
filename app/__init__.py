import os
from flask import Flask, flash, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json 
import cv2 
import os
app = Flask(__name__)

CORS(app)


UPLOAD_FOLDER = './test'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','JPG','PNG'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

res = {}
cal = 0

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
    print("sirve")
    for i in os.listdir('test'):
        img = image.load_img('test/'+i,target_size=(500,500))
        #plt.imshow(img)
        #plt.show()

        X = image.img_to_array(img)
        X = np.expand_dims(X,axis = 0)
        images = np.vstack([X]) 
        val = model.predict(images)
        js= model.evaluate(images)
        print(js)
        if(val == 0):
            foodd = 'pizza'
            cal = 1000
        else:
            foodd = 'pie'
            cal = 900
        res[i] = {
        "food": foodd,
        "accuracy": .80,
        "calories": cal,
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
