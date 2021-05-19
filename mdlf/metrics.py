class Metric:
    def __str__(self):
        return "Metric"

class Accuracy(Metric):
    def __call__(self, predicted, target):
        return (predicted == target).to(predicted.dtype).mean()

    def __str__(self):
        return super().__str__() + ": Accuracy"

class BinaryAccuracy(Metric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, predicted, target):
        pred = (predicted >= self.threshold)
        return (pred == target).to(predicted.dtype).mean()

    def __str__(self):
        return super().__str__() + ": Binary Accuracy"

class BinaryAccuracyTest(Metric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, predicted, target):
        pred = ((predicted+1)/2 >= self.threshold)
        return (pred == target).to(predicted.dtype).mean()

class test_3(Metric):
    def __call__(self, output, target):
        nb_test_errors = 0
        for n in range(target.size(0)):
            pred = output[n].max(0)[1].item()
            if target[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1
        return (100.0 *nb_test_errors / target.size(0))