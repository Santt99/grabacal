import os
from flask import Flask, flash, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
app = Flask(__name__)

CORS(app)


UPLOAD_FOLDER = './test'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
