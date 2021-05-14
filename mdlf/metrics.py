class Accuracy:
    def __call__(self, predicted, target):
        return (predicted == target).to(predicted.dtype).mean()

class BinaryAccuracy:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, predicted, target):
        pred = (predicted >= self.threshold)
        return (pred == target).to(predicted.dtype).mean()