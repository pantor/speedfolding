import numpy as np


class Max:
    """Returns the index of the maximum value in an array."""
    def __call__(self, x) -> int:
        return x.argmax()

    def disable(self, index, x):
        x.ravel()[index] = -np.inf

    def __repr__(self):
        return 'max'


class Top:
    """Returns the index of one of the highest N values with equal probability."""
    def __init__(self, n: int = 5, seed: int = None):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def __call__(self, x) -> int:
        return self.rng.choice(np.argpartition(x, -self.n, axis=None)[-self.n:])

    def disable(self, index, x):
        x.ravel()[index] = -np.inf

    def __repr__(self):
        return f'top{self.n}'
