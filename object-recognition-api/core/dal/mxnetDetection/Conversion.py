from core.models.Detection import BoundingBox, Detection


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


def serialize_bounding_box(bounding_box):
    return {'xmin': bounding_box.xmin,
            'ymin': bounding_box.ymin,
            'xmax': bounding_box.xmax,
            'ymax': bounding_box.ymax}


def serialize_detection(detection):
    return {'class': detection.classname,
            'score': detection.score,
            'boundingBox': serialize_bounding_box(detection.boundingBox)}


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
