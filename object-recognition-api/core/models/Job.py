from enum import Enum


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
