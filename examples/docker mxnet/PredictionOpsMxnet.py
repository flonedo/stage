import os
from gluoncv import data
from mxnet import gluon
import tempfile
import requests
import PIL.Image
from enum import Enum


def serialize_detection(detection):
    return {'class': detection.classname,
            'score': detection.score,
            'boundingBox': detection.boundingBox}


def serialize_image_detection(image_detection):
    dets = []
    for det in image_detection.detections:
        dets.append(serialize_detection(det))
    return {'clientId': image_detection.image.client_id,
            'uri': image_detection.image.uri,
            'detections': dets}


def serialize_image(image):
    return {'clientId': image.client_id,
            'uri': image.uri}

class Status(Enum):
    BUSY = 'BUSY'
    FAILED = 'FAILED'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'


class Job:
    def __init__(self, client_id, job_id, payload, status="BUSY"):
        self.client_id = client_id
        self.id = job_id
        self.payload = payload
        self.status = Status[status]


def deserialize_job(job):
    return Job(job['clientId'], job['id'], job['payload'], job['status'])


class Image:
    def __init__(self, client_id, uri):
        self.client_id = client_id
        self.uri = uri


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class Detection:
    def __init__(self, classname, score, bbox):
        self.classname = classname
        self.score = score
        self.boundingBox = bbox


class ImageDetection:
    def __init__(self, image):
        self.image = image
        self.detections = []

    @classmethod
    def from_detections_list(cls, image, detections):
        image_detection = ImageDetection(image)
        for detection in detections:
            image_detection.__add_detection(detection)
        return image_detection

    @classmethod
    def from_single_detection(cls, image, detection):
        image_detection = ImageDetection(image)
        image_detection.__add_detection(detection)
        return image_detection

    def __add_detection(self, detection):
        self.detections.append(detection)



class DetectionsFromNDArrays:
    def __init__(self, class_ids, classes, scores, bounding_boxes, scale):
        self.detections = []
        for i, score in enumerate(scores):
            score = float((scores[0][i]).asscalar())
            if score > 0:
                xmin = scale*(bounding_boxes[0][i][0]).asscalar()
                ymin = scale*(bounding_boxes[0][i][1]).asscalar()
                xmax = scale*(bounding_boxes[0][i][2]).asscalar()
                ymax = scale*(bounding_boxes[0][i][3]).asscalar()
                id = int((class_ids[0][i]).asscalar())
                class_name = classes[int(id)]
                detection = Detection(class_name, score, BoundingBox(xmin, ymin, xmax, ymax))
                self.detections.append(detection)

    def with_percentage_bounding_boxes(self, width, height):
        dets = []
        for det in self.detections:
            xmin_perc = float(det.boundingBox.xmin)/float(width)
            ymin_perc = float(det.boundingBox.ymin)/float(height)
            xmax_perc = float(det.boundingBox.xmax)/float(width)
            ymax_perc = float(det.boundingBox.ymax)/float(height)
            dets.append(Detection(det.classname, det.score, BoundingBox(xmin_perc, ymin_perc, xmax_perc, ymax_perc)))
        self.detections = dets
        return self.detections


class Network:
    def __init__(self, network_model, network_params, synset):
        self.model = network_model
        self.parameters = network_params
        self.synset = synset



class PredictionOpsMxnet:
    def __init__(self, path_to_models, network_name='network.json', parameters_name='parameters.params',
                 synset_name='synset.txt', resize=416):
        self.path_to_models = path_to_models
        self.network_name = network_name
        self.parameters_name = parameters_name
        self.synset_name = synset_name
        self.resize = resize

    def __get_latest_model(self, client_id):
        models = os.listdir(os.path.join(self.path_to_models, client_id))
        models.sort(reverse=True)
        latest_model = os.path.join(self.path_to_models, client_id, models[0])
        net = os.path.join(latest_model, self.network_name)
        params = os.path.join(latest_model, self.parameters_name)
        synset = os.path.join(latest_model, self.synset_name)
        return Network(net, params, synset)

    @staticmethod
    def __get_dataset_class_names(synset_path):
        classes = []
        with open(synset_path) as syn:
            for line in syn:
                classes.append(line.rstrip())
        return classes

    @staticmethod
    def __get_bounding_box_scale(resize, width, height):
        shortest = min(width, height)
        return 1/(resize/shortest)

    def predict(self, job):
        model = self.__get_latest_model(job.client_id)
        net = gluon.nn.SymbolBlock.imports(model.model, ['data'], model.parameters)
        classes = self.__get_dataset_class_names(model.synset)
        image = (requests.get(job.payload, stream=True)).raw.read()
        # on windows a temp file must be deleted manually, otherwise it won't be accessible outside the method creating it
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        try:
            temp_image.write(image)
            original_width, original_height = PIL.Image.open(temp_image).size
            temp_image.close()
            image_ndarray, resized_img = data.transforms.presets.yolo.load_test(temp_image.name, short=self.resize)
        finally:
            # manual deletion of temp file
            os.remove(temp_image.name)
        class_ids, scores, bounding_boxes = net(image_ndarray)
        resize_scale = self.__get_bounding_box_scale(self.resize, original_width, original_height)
        detections = DetectionsFromNDArrays(class_ids, classes, scores, bounding_boxes, resize_scale)\
            .with_percentage_bounding_boxes(original_width, original_height)
        return ImageDetection.from_detections_list(Image(job.client_id, job.payload), detections)
