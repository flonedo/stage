import unittest
import os
from core.dal.mxnetDetection.PredictionOpsMxnet import PredictionOpsMxnet
from core.models.Job import Job
from core.dal.mxnetDetection.Conversion import serialize_image_detection
from core.models.Detection import ImageDetection


class PredictionOpsMxnetSpec(unittest.TestCase):
    path_to_models = os.environ.get('NETWORK_MODELS_PATH', None)
    ops = PredictionOpsMxnet(path_to_models)

    def test_predict(self):
        client_id = 'test_id'
        job_id = 'test_job_id'
        uri = 'https://hips.hearstapps.com/roa.h-cdn.co/assets/16/23/1465611331-alfa-romeo-giulia-030-1-1.jpg'
        status = 'RUNNING'
        job = Job(client_id, job_id, uri, status)
        image_prediction = self.ops.predict(job)
        print(serialize_image_detection(image_prediction))
        assert image_prediction.image.client_id == job.client_id
