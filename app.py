#!/usr/bin/python3
import numpy as np
from flask import Flask, request, Response, abort
from flask_cors import CORS
from tensorflow import keras
import json

def pad_image(img, pad_t, pad_r, pad_b, pad_l):
    """Add padding of zeroes to an image.
    Add padding to an array image.
    :param img:
    :param pad_t:
    :param pad_r:
    :param pad_b:
    :param pad_l:
    """
    height, width = img.shape

    height = int(height)

    width = int(width)

    # Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    # Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    # Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    # Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img

def center_image(img):
    """Return a centered image.
    :param img:
    """
    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[y1:y2, x1:x2]

    zero_axis_fill = (img.shape[0] - cropped_image.shape[0])
    one_axis_fill = (img.shape[1] - cropped_image.shape[1])

    top = zero_axis_fill / 2
    bottom = zero_axis_fill - top
    left = one_axis_fill / 2
    right = one_axis_fill - left

    padded_image = pad_image(cropped_image, int(top), int(left), int(bottom), int(right))

    return padded_image

app = Flask(__name__)
CORS(app)
# Load mnist model
model = keras.models.load_model('model')

# /predict endpoint
@app.post('/predict')
def predict():
    req = request.get_json()['pixels']
    pixels_c = center_image(np.reshape(req, (28, 28)))
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
