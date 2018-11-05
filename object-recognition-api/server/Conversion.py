from core.models.Image import Image
from core.models.Job import Status


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


def deserialize_image(image):
    return Image(image['clientId'], image['uri'])


def serialize_job(job):
    if job.status == Status.RUNNING:
        status = Status.BUSY
    else:
        status = job.status
    return {'clientId': job.client_id,
            'id': job.id,
            'image': {
                'uri': job.payload
            },
            'url': get_url(job.client_id, job.id),
            'status': status.value}


def serialize_error(error):
    return {'code': error.code, 'message': error.message, 'errors': error.errors}


def get_url(client_id, job_id):
    return str('api/prediction/images/' + client_id + '/' + job_id)

