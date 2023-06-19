#!/usr/bin/python3
import numpy as np
from flask import Flask, request, Response, abort
from flask_cors import CORS
from tensorflow import keras
# import json
# import scipy as sp
# from scipy.ndimage import zoom


# def clipped_zoom(img, zoom_factor, **kwargs):

#     h, w = img.shape[:2]

#     # For multichannel images we don't want to apply the zoom factor to the RGB
#     # dimension, so instead we create a tuple of zoom factors, one per array
#     # dimension, with 1's for any trailing dimensions after the width and height.
#     zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

#     # Zooming out
#     if zoom_factor < 1:

#         # Bounding box of the zoomed-out image within the output array
#         zh = int(np.round(h * zoom_factor))
#         zw = int(np.round(w * zoom_factor))
#         top = (h - zh) // 2
#         left = (w - zw) // 2

#         # Zero-padding
#         out = np.zeros_like(img)
#         out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

#     # Zooming in
#     elif zoom_factor > 1:

#         # Bounding box of the zoomed-in region within the input array
#         zh = int(np.round(h / zoom_factor))
#         zw = int(np.round(w / zoom_factor))
#         top = (h - zh) // 2
#         left = (w - zw) // 2

#         out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

#         # `out` might still be slightly larger than `img` due to rounding, so
#         # trim off any extra pixels at the edges
#         trim_top = ((out.shape[0] - h) // 2)
#         trim_left = ((out.shape[1] - w) // 2)
#         out = out[trim_top:trim_top+h, trim_left:trim_left+w]

#     # If zoom_factor == 1, just return the input array
#     else:
#         out = img
#     return out

# def recenter(arr):
#     slicing = sp.ndimage.find_objects(arr != 0, max_label=1)[0]
#     center_slicing = tuple(
#         slice((dim - sl.stop + sl.start) // 2, (dim + sl.stop - sl.start) // 2)
#         for sl, dim in zip(slicing, arr.shape))
#     result = np.zeros_like(arr)
#     result[center_slicing] = arr[slicing]
#     return result

app = Flask(__name__)
CORS(app)
# Load mnist model
model = keras.models.load_model('model')

# /predict endpoint
@app.post('/predict')
def predict():
    req = request.get_json()['pixels']
    # pixels_c = recenter(np.reshape(req, (28, 28)))
    pixels = np.array(req)
    pixels_f = np.reshape(pixels, (1, 28, 28, 1))
    # pixels_f = clipped_zoom(pixels_c, 1.1)
    if pixels_f is None:
       abort(400)
    prediction = model.predict([pixels_f])
    res = prediction[0].tolist()
    print('pred:', res)
    return {"result": res}

# Run the app
if __name__ == "__main__":
 app.debug = True
 app.run(host="0.0.0.0")
