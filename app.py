#!/usr/bin/python3
import numpy as np
from flask import Flask, request, Response, abort
from flask_cors import CORS
from tensorflow import keras
import json

app = Flask(__name__)
CORS(app)
# Load mnist model
model = keras.models.load_model('model')

# /predict endpoint
@app.post('/predict')
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
 app.debug = True
 app.run(host="0.0.0.0")
