# import flask
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
import re

# below line remove the dependency on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
import logging

app = Flask(__name__, template_folder='templates')
# loaded_model = joblib.load('/home/ashritagandhari/embryopredict/Embryo_Prediction_Model.pkl')
my_dir = os.path.dirname(__file__)
upload_folder = os.path.join(my_dir, 'static/upload_folder')
model_file_path = os.path.join(my_dir, 'Embryo_Prediction_Model.h5')
loaded_model = keras.models.load_model(model_file_path)

img_size = 256


def preprocess(img):
    img = np.array(img)
    inputimage1 = cv2.resize(img, (256, 256))
    logging.warning('input image resized TO : %s ', inputimage1.shape)
    # Normalizing the image
    inputimage2 = inputimage1 / 255.0
    inputimage3 = inputimage2.reshape((-1, 256, 256, 3))
    return inputimage3


@app.route("/")
def root():
    return render_template('index.html')


@app.route("/make_prediction", methods=['POST'])
def make_prediction():
    logging.debug('THIS IS A SAMPLE DEBUG MESSAGE')
    print('THIS IS A SAMPLE DEBUG MESSAGE')
    logging.warning('Request Method %s', request.method)
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    dataBytesIO.seek(0)
    image = Image.open(dataBytesIO)
    test_image = preprocess(image)

    logging.warning('Submitting for prediction')
    prediction = loaded_model.predict(test_image)
    logging.warning('prediction received :%s ', prediction)

    pred_class = np.argmax(prediction) + 1

    if pred_class == 1:
        msg = "This Embryo is at the 'Degenerate' Embryo stage. Embryo failed to develop."
    elif pred_class == 2:
        msg = "This Embryo is at the 'Morula' stage. More than 50% of the embryo has undergone compaction."
    elif pred_class == 3:
        msg = "This Embryo is at the 'Early Blastocyst' stage. Balstocoele less than the volume of the embryo."
    elif pred_class == 4:
        msg = "This Embryo is at the 'Full Blastocyst' stage. Blastocoele completely filling embryo."
    elif pred_class == 5:
        msg = "This Embryo is at the 'Hatched Blastocyst' stage. Blastocyst completely hatched."

    accuracy = float(np.max(prediction, axis=1)[0])

    response = {'prediction': {'result': msg, 'accuracy': accuracy}}
    return jsonify(response)


def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


if __name__ == '__main__':
    app.run(debug=True)
