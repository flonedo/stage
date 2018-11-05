import connexion
import os
from core.dal.mongodb.JobOpsMongo import JobOpsMongo
from core.dal.mxnetDetection.PredictionOpsMxnet import PredictionOpsMxnet
from core.services.JobService import JobService
from core.models.Job import Job
from core.models.Detection import ImageDetection
from test.core.dal.mxnetDetection.PredictionOpsMxnetSpec import PredictionOpsMxnetSpec
from core.models.Error import Error400
import server.Conversion
import pymongo
import json


def insertPrediction(prediction):
    image = server.Conversion.deserialize_image(prediction)
    if image.client_id.isalnum():
        job = jobService.insert(image)
        location_url = server.Conversion.get_url(job.client_id, job.id)
        return server.Conversion.serialize_job(job), 202, {'Location': location_url}
    else:
        return server.Conversion.serialize_error(Error400()), 400

def getPrediction(clientId, jobId):
    if clientId.isalnum():
        prediction = jobService.find(clientId, jobId)
        location_url = server.Conversion.get_url(clientId, jobId)
        if type(prediction) is ImageDetection:
            return server.Conversion.serialize_image_detection(prediction), 200, {'Location': location_url}
        if type(prediction) is Job:
            return server.Conversion.serialize_job(prediction), 202, {'Location': location_url}
        else:
            return server.Conversion.serialize_error(prediction), prediction.code
    else:
        return server.Conversion.serialize_error(Error400()), 400


def deletePrediction(clientId, jobId):
    if clientId.isalnum():
        del_result = jobService.remove(clientId, jobId)
        return server.Conversion.serialize_image(del_result), 200
    else:
        return server.Conversion.serialize_error(Error400()), 400


app = connexion.App(__name__)

client_path = os.environ.get('MONGODB_URI', None)
network_models_path = os.environ.get('NETWORK_MODELS_PATH', None)
with open(os.environ.get('APP_PATH', None) + "init.json", "r") as init:
    config = json.load(init)
app.add_api(os.environ.get('APP_PATH', None) + "open-api.yaml", validate_responses=True)
database_name = config['database_name']
collection_name = config['collection_name']
network_model_name = config['network_model_name']
network_params_name = config['network_params_name']
synset_name = config['network_synset_name']
image_resize_dimension = config['image_resize_dimension']
client = pymongo.MongoClient(client_path)
job_ops = JobOpsMongo(client, database_name, collection_name)
prediction_ops = PredictionOpsMxnet(network_models_path, network_model_name, network_params_name,
                                    synset_name, image_resize_dimension)
jobService = JobService(job_ops)
PredictionOpsMxnetSpec().test_predict()

if __name__ == '__main__':
    app.run(port=8080)
