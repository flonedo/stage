# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import flask
from PredictionOpsMxnet import PredictionOpsMxnet, deserialize_job, serialize_image_detection

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    def __init__(self):
        self.ops = PredictionOpsMxnet(model_path)

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (path to jpg image): The data on which to do the predictions."""
        return cls.ops.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    if flask.request.content_type == 'json':
        data = flask.request.data.decode('utf-8')
        job = deserialize_job(data)
    else:
        return flask.Response(response='This predictor only supports json data', status=415, mimetype='text/plain')
    # Do the prediction
    predictions = serialize_image_detection(ScoringService.predict(job))
    return flask.Response(response=predictions, status=200, mimetype='text/csv')
