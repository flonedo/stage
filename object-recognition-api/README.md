# Object detection API YOLO v3



## Installation
We implemeted our detection server as a *Flask* Python server application. To track to our server application what we declared in our OpenAPI we used the *Connexion* framework; this framework automatically connects the *operationIds* we declared in our API to the functions in our Flask server app and validates all of the JSON data exchange, doing most of the hard work for us.
To implement our jobs queue, we used a *MongoDB* collection.
We provide a *requirements.txt* file to install all the needed dependencies to run our code; however if you want to manually install the requirements you can do it from *pip* as follows:

```sh
$ pip install connexion
$ pip install pymongo
$ pip install requests
$ pip install gluoncv mxnet
```
Note that you only need to install Mxnet Framework and GluonCV toolkit if you want to run inference code.
 Our code is tested on Python 3.6.

 ### Docker
 if you want to run our application out-of-the-box you can build the provided Dockerfile and work from there.

## Application overview
Our *Flask* ([Note 1](.#Note-1:)) Python server responds to HTTP requests. Currently, the following three requests are implemented:
- __POST__: the HTTP POST request defined in the OpenAPI is connected to the *insertPrediction(prediction)* function; if successful, the function inserts a new inference job in our queue and returns the client the URL where to check for the inference result ([Note 2](.#Note-2:)).
- __GET__: the HTTP GET request in the OpenAPI is connected to the *getPrediction(clientId, jobId)* function; if the inference result is available it returns it, otherwise it returns the URL where to check for it later.
- __DELETE__: the HTTP DELETE request in the OpenApi is connected to the *deletePrediction(clientId, jobId) function; if the job to delete is found in the queue, rìthe function deletes it.

For further details about our HTTP requests please refer to our OpenAPI.
To decouple our server interface from its implementation, we call our methods through generic Services.
###### __Note 1:__
Since our application runs on top of *Connexion* framework instead of being a pure *Flask* app, we couldn't custom handle some errors (e.g. 500 server error).
###### __Note 2:__
Our application handles inference asynchronously with polling. After a HTTP POST request is received, a corresponding job is created and inserted in our jobs queue; only at some later point in time the actual inference will be run and its result put in our database. No callbacks are sent, so it is up to the client to periodically check whether the prediction has been executed yet.
![alt text](D:/data/object-recognition-api/PredictionService.png "Flux diagram")

### Models, serialization and deserialization
Within our application requests parameters are handled as Python objects. However, both our clients and our database expect data in JSON format so we implemented serialization and deserialization as needed for each object model.
You can find the definition of each JSON object model in our OpenAPI.

### Jobs queue
Our inference jobs queue consists of a MongoDB collection; operations to connect to it and manage jobs are defined in the *JobOpsMongo*.*py* module. The class creates an object for database operations and provides following methods:
- __enqueue(job)__: inserts a new job in the queue. The input parameter *job* must be a valid *Job* class instance.
- __dequeue()__: extracts the oldest job from the queue and returns it as a *Job* class instance.
- __find(clientId, jobId)__: finds the job in the queue and retuns it as a *Job* class instance.
- __remove(clientId, jobId)__: removes the job from the queue and returns is as a *Job* class instance.

To better decouple our server interface from its implementation, the operations defined in the *JobOpsMongo*.*py* module are called through a generic *JobService*.*py* module. The *JobService* is instantiated with *JobOpsMongo* operations when the application starts.

### Inference
We provide a *PredictionService* for inference ([Note 3](.#Note-3:)); its operations are actually implemented in the *PredictionOpsMxnet*.*py* module to increase decoupling.
The *PredictionOpsMxnet*.*py* module creates an object for detection that defines a method for detection:
- __predict(job)__: takes an input *Job* class object and returns an *ImageDetection* class object containing the detection results.
###### Note 3:
As of now the queue consumer that should call the *PredictionService* for inference has not been implemented yet; thus the functionality has only been used by unit tests so far.
If you want to get accustomed with the inference process we suggest experimenting a little with the *detect_yolo3.py* script in the *object-recognition-demo* project.




