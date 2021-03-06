from abc import abstractmethod
import numpy as np


class SimilarityMeasure(object):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ChiSquareSimilarity(SimilarityMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        den = x + y
        num = (x-y)**2
        num = num[den != 0]
        den = den[den != 0]
        distance = 0.5 * np.sum(np.divide(num, den))
        return 1 / (1 + distance)


class EuclideanSimilarity(SimilarityMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        distance = np.sqrt(np.sum((x-y)**2))
        return 1 / (1 + distance)


class CosineSimilarity(SimilarityMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        return np.sum(x*y) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))