import pymongo
from pymongo import ReturnDocument
from core.dal.mongodb.Conversion import serialize_job_ids, serialize_job, deserialize_job


class JobOpsMongo:
    def __init__(self, mongoClient, databaseName, collectionName):
        self.client = mongoClient
        self.databaseName = databaseName
        self.collectionName = collectionName
        self.database = self.client[self.databaseName]
        self.collection = self.database[self.collectionName]

    def enqueue(self, job):
        self.collection.insert_one(serialize_job(job))
        return job

    def dequeue(self):
        job = self.collection.find_one_and_update({"status": "BUSY"}, {'$set': {"status": "RUNNING"}},
                                                  sort=[('_id', pymongo.ASCENDING)],
                                                  return_document=ReturnDocument.AFTER)
        if job is not None:
            return deserialize_job(job)
        else:
            return job

    def find(self, clientId, jobId):
        job = self.collection.find_one(serialize_job_ids(clientId, jobId))
        if job is not None:
            return deserialize_job(job)
        else:
            return job

    def remove(self, clientId, jobId):
        deleted_job = self.collection.find_one_and_delete(serialize_job_ids(clientId, jobId))
        return deserialize_job(deleted_job)
