#!/usr/bin/python3
import numpy as np
from flask import Flask, request, Response, abort
from flask_cors import CORS
from tensorflow import keras
import json
import scipy as sp

def recenter(arr):
    slicing = sp.ndimage.find_objects(arr != 0, max_label=1)[0]
    center_slicing = tuple(
        slice((dim - sl.stop + sl.start) // 2, (dim + sl.stop - sl.start) // 2)
        for sl, dim in zip(slicing, arr.shape))
    result = np.zeros_like(arr)
    result[center_slicing] = arr[slicing]
    return result

app = Flask(__name__)
CORS(app)
# Load mnist model
model = keras.models.load_model('model')

# /predict endpoint
@app.post('/predict')
def predict():
    req = request.get_json()['pixels']
    # pixels_c = recenter(np.reshape(req, (28, 28)))
    pixels_c = recenter(req)
    pixels = np.reshape(pixels_c, (1, 28, 28, 1))
    if pixels is None:
       abort(400)
    prediction = model.predict([pixels])
    res = prediction[0].tolist()
    print('pred:', res)
    return {"result": res}

# Run the app
if __name__ == "__main__":
 app.debug = True
 app.run(host="0.0.0.0")
