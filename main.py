import os
import numpy as np
import requests
from flask import Flask, request
from flask_restful import Resource, Api

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
PRE_SCALE = float(os.getenv("PRE_SCALE", 1))
# POST_SCALE = float(os.getenv("POST_SCALE", 1)) #TODO: apply after squash, if squash
SQUASH = os.getenv("SQUASH", "").lower() in (True, 'true') #default to false

in_size = np.prod(INPUT_SHAPE)
out_size = np.prod(OUTPUT_SHAPE)
#xavier-ish, just not truncated
stddev = np.sqrt(2 / (in_size + out_size))

weights = np.random.normal(0, stddev, [in_size, out_size])
weights = np.where(
    np.random.random(size=weights.shape) < DENSITY,
    weights,
    np.zeros_like(weights)
)

def project(x):
    x = x.flatten()
    x = np.matmul(x, weights)
    x = np.reshape(x, OUTPUT_SHAPE)
    x = x * PRE_SCALE
    if SQUASH:
        x = np.tanh(x)
    return x

class Project(Resource):
    def post(self):
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
