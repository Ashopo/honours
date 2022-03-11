import autograd.numpy as np
from typing import Callable
import pybenchfunction.function as bench

class ObjectiveFunction:

    def __init__(self, func: Callable, **kwargs):
        self.__dict__.update(kwargs)
        self.func = func
    
    def eval(self, *args):
        """
        Evaluates the objective function at a specified coordinate in arbitrary dimensions.
        """
        x = np.array(args)
        return self.func(self, x)

    # Functions available
    def gaussian_wells(self, x: np.array):
        """
        Gaussian wells of differing depths, centers and standard deviations.
        Accepts arbitrary dimension coordinate input.
        
        x: a 2-vector of coordinates
        centers: list of 2-vector coordinates
        depths: list of scalar depths
        sigmas: list of scalar standard deviations
        """

        if len(self.centers) != len(self.depths) or len(self.depths) != len(self.sigmas) or len(self.centers) != len(self.sigmas):
            raise ValueError('Mismatched Gaussian well parameter lists.')

        z = 0
        for c, d, s in zip(self.centers, self.depths, self.sigmas):
            v = x - c
            z += d/(2*np.pi*s**2) * np.exp(-np.dot(v,v.T)/(2*s**2))

        return z
    
    def sse(self, x: np.array):
        """
        """
        pass
