import numpy as np


class FakeNetwork:
	def __init__(self, n_classes: int, learn_rate = 1.64 / 512, momentum = 0.85, l2 = 0.0):
		self.n_classes = n_classes

	def get_test_confusion_matrix(self) -> np.ndarray:
		return np.zeros((self.n_classes, self.n_classes))
