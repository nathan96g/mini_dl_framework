class Metric:
    def __str__(self):
        return "Metric"

class Accuracy(Metric):
    """
    This metric simply test the equality between the predictions and
    the targets.
    """

    def __call__(self, predicted, target):
        return (predicted == target).to(predicted.dtype).mean()

    def __str__(self):
        return super().__str__() + ": Accuracy"

class BinaryAccuracy(Metric):
    """
    This metric classify the predictions in two classes
    using a threshold. Then it compares these predicted
    classes and the targets.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, predicted, target):
        pred = (predicted >= self.threshold)
        return (pred == target).to(predicted.dtype).mean()

    def __str__(self):
        return super().__str__() + ": Binary Accuracy"