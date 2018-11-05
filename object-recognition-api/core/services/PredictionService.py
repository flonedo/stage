class PredictionService:
    def __init__(self, prediction_ops):
        self.predictionOps = prediction_ops

    def predict(self, job):
        image_prediction = self.predictionOps.predict(job)
        return image_prediction
