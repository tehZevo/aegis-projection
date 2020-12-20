import os
import numpy as np
import requests
from flask import Flask, request
from flask_restful import Resource, Api
from keras.layers import Dense
from keras.models import Sequential, load_model

from nd_to_json import nd_to_json, json_to_nd

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

PORT = os.getenv("PORT", 80)
SOURCE_URL = os.getenv("SOURCE_URL")
INPUT_SHAPE = os.getenv("INPUT_SHAPE").split()
INPUT_SHAPE = [int(i) for i in INPUT_SHAPE]
OUTPUT_SHAPE = os.getenv("OUTPUT_SHAPE").split()
OUTPUT_SHAPE = [int(i) for i in OUTPUT_SHAPE]
DENSITY = float(os.getenv("DENSITY", 0.1))
INPUT_SCALE = float(os.getenv("INPUT_SCALE", 1))
OUTPUT_SCALE = float(os.getenv("OUTPUT_SCALE", 1))
ACTIVATION = os.getenv("ACTIVATION", None)

MODEL_PATH = os.getenv("MODEL_PATH", "models/projection")

in_size = np.prod(INPUT_SHAPE)
out_size = np.prod(OUTPUT_SHAPE)

try:
    model = load_model(MODEL_PATH)
except:
    print('"{}" not found. Creating new model.'.format(MODEL_PATH))
    model = Sequential([
        Dense(out_size, activation=ACTIVATION, input_shape=[in_size])
    ])
    print([x.shape for x in model.get_weights()])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH, save_format="h5")#TODO: use tf in the future

model.summary()

def project(x):
    x = x.flatten()
    x = x * INPUT_SCALE
    x = model.predict(np.expand_dims(x, 0))[0]
    x = x * OUTPUT_SCALE
    x = np.reshape(x, OUTPUT_SHAPE)

    return x

class Project(Resource):
    def post(self):
        #TODO: use ppcl for this?
        r = requests.post(SOURCE_URL)
        r.raise_for_status()
        x = json_to_nd(r.json())
        x = project(x)
        x = nd_to_json(x)

        return x

app = Flask(__name__)
api = Api(app)

api.add_resource(Project, "/")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
