#!/usr/bin/python3
# import cv2
import numpy as np
from flask import Flask, request, Response, abort
# from joblib import dump, load
from tensorflow import keras
import json

application = Flask(__name__)
# Load mnist model
model = keras.models.load_model('src\model')

# /predict endpoint
@application.post('/predict')
def predict():
    req = request.get_json()['pixels']
    pixels = np.reshape(req, (1, 28, 28, 1))
    if pixels is None:
       abort(400)
    prediction = model.predict([pixels])
    res = prediction[0].tolist()
    print('pred:', res)
    return {"result": res}

# Run the app
if __name__ == "__main__":
 application.debug = True
 application.run(host="0.0.0.0")
