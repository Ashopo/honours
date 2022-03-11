import autograd.numpy as np
from autograd import grad
from typing import Callable

class Optimiser:

    def __init__(self, func):
        self._func = func
        self.func = func.eval
        self.path = []
    
    def get_path(self):
        """
        Returns the optimiser walker path with the corresponding objective 
        function values.
        """

        coords = [list(_) for _ in list(zip(*self.path))]

        return coords, [float(self.func(x)) for x in self.path]

    def update_params(self, **kwargs):
        self.__dict__.update(kwargs)
    
class GradientDescent(Optimiser):

    def __init__(self, func: Callable, eta: float, n_epochs: int):
        super().__init__(func)
        self.grad = grad(self.func)
        self.eta = eta
        self.n_epochs = n_epochs

    def get_update_vector(self, x: np.array):
        """
        Returns the canonical gradient descent update vector.
        """
        return self.eta * self.grad(x)

    def optimise(self, start: np.array, restart=True):
        """
        Find minimum of objective function from a specified starting point using
        the gradient descent algorithm.

        Returns the list of coordinates visited.
        """

        if restart:
            self.path = []

        self.path.append(start)
        x = start

        for i in range(self.n_epochs):
            x = x - self.get_update_vector(x)
            self.path.append(x)

        return self.get_path()

class StochasticGradientDescent(GradientDescent):
    """
    Implements batch gradient descent. 
    Set batch_size = 1 for 'true' stochastic gradient descent
    """

    def __init__(self, func: Callable, eta: float, n_epochs: int, batch_size: int):
        super().__init__(func, eta, n_epochs)
        self.batch_size = batch_size
    
    # @ Override canonical gradient descent class
    def get_update_vector(self, x: np.array):
        """
        TODO:
            Implement a summation based objective function e.g. SSE
                - Need to be able to control which examples the function uses for
                  evaluation.
                - Randomly sample which examples to evaluate the gradient on.
            SGD doesn't make sense for gaussian wells/non sum functions.
            Consider a pytorch pivot.
            Pytorch has momentum and ADAM already implemented, as well as NAG.
        """
        pass
