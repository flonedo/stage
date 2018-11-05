import os
from gluoncv import data
from mxnet import gluon
from core.dal.mxnetDetection.Network import Network
import tempfile
import requests
from core.dal.mxnetDetection.Conversion import DetectionsFromNDArrays
from core.models.Detection import ImageDetection
from core.models.Image import Image
import PIL.Image


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
