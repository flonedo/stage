from core.models.Job import Job


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


def serialize_job(job):
    return {'clientId': job.client_id,
            'id': job.id,
            'payload': job.payload,
            'status': job.status.value}


def serialize_job_ids(client_id, job_id):
    return {'clientId': client_id,
            'id': job_id}


def deserialize_job(job):
    return Job(job['clientId'], job['id'], job['payload'], job['status'])

