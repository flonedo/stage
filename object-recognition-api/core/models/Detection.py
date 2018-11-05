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
