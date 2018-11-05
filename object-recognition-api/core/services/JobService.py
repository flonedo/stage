from core.models.Job import Job
from core.models.Image import Image
import hashlib


class JobService:
    def __init__(self, jobOps):
        self.jobOps = jobOps

    @classmethod
    def __calculate_job_id(cls, image):
        return str(hashlib.md5(image.uri.encode()).hexdigest())

    def insert(self, image):
        job_id = self.__calculate_job_id(image)
        job = Job(image.client_id, job_id, image.uri)
        self.jobOps.enqueue(job)
        return job

    def find(self, clientId, jobId):
        job = self.jobOps.find(clientId, jobId)
        return job

    def remove(self, clientId, jobId):
        deleted_job = self.jobOps.remove(clientId, jobId)
        return Image(deleted_job.client_id, deleted_job.payload)
