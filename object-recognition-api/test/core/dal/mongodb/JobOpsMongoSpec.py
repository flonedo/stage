import unittest
import pymongo as pymongo
from core.dal.mongodb.JobOpsMongo import JobOpsMongo
from core.models.Job import Job, Status
from core.dal.mongodb.Conversion import serialize_job


class JobOpsMongoSpec(unittest.TestCase):
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    database_name = "predictions"
    collection_name = "jobs"
    ops = JobOpsMongo(client, database_name, collection_name)

    def test_enqueue(self):
        job = Job("test_client_id", "test_id", "imageUri", "BUSY")
        inserted_job = self.ops.enqueue(job)
        self.assertEqual(serialize_job(job), serialize_job(inserted_job))

    def test_dequeue(self):
        job = self.ops.dequeue()
        if job is not None:
            self.assertTrue(job.status == Status.RUNNING)

    def test_remove(self):
        job = Job("test_del_client_id", "test_del_id", "test_del_image", "BUSY")
        inserted_job_id = self.ops.collection.insert_one(serialize_job(job)).inserted_id
        deleted_job = self.ops.remove("test_del_client_id", "test_del_id")
        in_collection = self.ops.collection.find_one({"_id": inserted_job_id})
        self.assertTrue(in_collection is None)


if __name__ == '__main__':
    unittest.main()
