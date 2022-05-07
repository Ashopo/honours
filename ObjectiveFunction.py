import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim.optimizer import Optimizer
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Callable
import helper_funcs as hf

class OptimisationProblem():

    def __init__(
        self, 
        func: nn.Module, 
        opt: Optimizer,
        n_epochs: int,
        metric: Callable = None
    ):
        self.func = func
        self.opt = opt
        self.n_epochs = n_epochs
        self.losses, self.params, self.preds = None, None, None
        self.metric = metric

        if hasattr(opt, '_metric'):
            if metric is not None:
                raise ValueError("Given optimiser already has a custom metric. Cannot specify another metric.")
            self.metric = opt._metric

    def run(self):
        func = self.func
        opt = self.opt
        losses, params, preds = [], [], []
        params.append(list(func.parameters())[0].detach().clone().numpy())
        #params.append(list(func.parameters())[0].data.detach().clone().numpy())

        for i in range(self.n_epochs):
            pred = func()
            loss = self.metric(pred) if self.metric else pred
            opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parameters, max_norm) # added this 21/04/2022 - apparently stops exploding gradients
            opt.step()
            losses.append(float(loss))
            #print(list(func.parameters())[0])
            params.append(list(func.parameters())[0].detach().clone().numpy())
            # params.append(list(func.parameters())[0].data.detach().clone().numpy()) # WTF CLONE CHANGES THINGS 21/04
            # apparently this is because it changes what is saved in place - diff tensors in same place in memory if you detach
            # want to clone if you want to save the weights i.e. parameters at each step 21/04
            preds.append(float(pred))
        
        params = [list(_) for _ in list(zip(*params))]

        self.losses = losses
        self.params = params
        self.preds = preds

        return losses, params, preds

    def get_results(self):
        return self.losses, self.params, self.preds
    
    def visualise(
        self, 
        xlim: Tuple,
        ylim: Tuple,
        res: float,
        render: str = None
    ):
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(self.preds)), y=self.preds))
        fig1.update_layout(
            autosize=False,
            width=800,
            height=800
        )

        if render is None: return fig1

        if render == "3d":
            X = np.arange(xlim[0], xlim[1], res)
            Y = np.arange(ylim[0], ylim[1], res)
            grid = [np.array(_) for _ in itertools.product(X,Y)]
            Z = [float(self.func.forward(torch.Tensor(_))) for _ in grid]
            Z = np.array(Z).reshape(len(X), len(Y)).T

            fig2 = go.Figure()
            fig2.add_trace(go.Surface(x=X, y=Y, z=Z))
            fig2.add_trace(go.Scatter3d(x=self.params[0], 
                                        y=self.params[1], 
                                        z=self.preds,
                                        mode='markers',
                                        marker=dict(
                                            color=np.arange(0,len(self.preds))),
                                            opacity=0.5
                                        )
            )

            fig2.update_layout(
                autosize=False,
                width=800,
                height=800
            )

            return fig1, fig2

        elif render == "contour":
            fig = make_subplots(rows=1,cols=2)
            X = np.arange(xlim[0], xlim[1], res)
            Y = np.arange(ylim[0], ylim[1], res)
            grid = [np.array(_) for _ in itertools.product(X,Y)]
            Z = [float(self.func.forward(torch.Tensor(_))) for _ in grid]
            Z = np.array(Z).reshape(len(X), len(Y)).T

            fig.add_trace(fig1.data[0], row=1, col=1)
            fig.add_trace(
                go.Contour(x=X, y=Y, z=Z),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=self.params[0], 
                           y=self.params[1],
                           text=list(zip(self.preds, list(np.arange(0, len(self.preds)))))),
                row=1, col=2
            )
            fig.update_layout(
                autosize=False,
                width=1600,
                height=800
            )
            
            return fig

        else:
            raise ValueError("""
                            Required render option does not exist. Allowed renders include:
                                        - 3d
                                        - contour
                             """)

    def tmsd_analysis(self, max_tau_ratio):
        
        if self.params is None:
            raise NameError("Must run the optimisation problem before finding the optimiser MSD. Call .run() before .get_msd()")
        
        return hf.tmsd_analysis(self.params, max_tau_ratio)
    
class PeriodicObjectiveFunction(nn.Module):

    def __init__(self, bounds=None):
        super().__init__()
        self.bounds = bounds
    
    def pbc(self, X):

        if X >= 0:
            q = X/self.bounds
            q = q + 0.5
            return X - self.bounds * np.floor(q)
        if X < 0:
            q = X/self.bounds
            q = q - 0.5
            return X - self.bounds * np.ceil(q)


    def apply_period(self, X):

        if self.bounds is None:
            return X
        X.data = X.data.apply_(self.pbc)
        
        return X

class GaussianWellModel(PeriodicObjectiveFunction):

    def __init__(self, start, bounds=None):
        
        super().__init__(bounds)
        weights = torch.Tensor(start)
        self.weights = nn.Parameter(weights)
        self.centers = [torch.Tensor([-6,4]), torch.Tensor([5,-2]), torch.Tensor([-4,-4])]
        self.depths = [torch.Tensor([-0.05]), torch.Tensor([-0.05]), torch.Tensor([-0.04])]
        self.sigmas = [torch.Tensor([3.5]), torch.Tensor([5]), torch.Tensor([5.5])]
        
    def forward(self, X=None):
        """
        Implement function to be optimised.
        """

        if X is None:
            X = self.weights
        
        X = self.apply_period(X)
        z = 0
        for c, d, s in zip(self.centers, self.depths, self.sigmas):
            v = X - c
            z += d/(2*np.pi*s**2) * torch.exp(-torch.dot(v,v.T)/(2*s**2))

        return z

class AlpineN1(PeriodicObjectiveFunction):
    """
    Bumpy function with many local minima.
    Overall function slopes inward, so momentum algorithms will 
    accumulate a center-directed gradient if they stay on one side
    of the global minimum.

    Global minimum at x* = 0, f(x*) = 0.
    """
    def __init__(self, start, bounds=None):
        
        super().__init__(bounds)
        weights = torch.Tensor(start)
        self.weights = nn.Parameter(weights)
        
    def forward(self, X=None):

        if X is None:
            X = self.weights

        X = self.apply_period(X)
        z = torch.abs(X*torch.sin(X) + 0.1*X)
        return torch.sum(z)

class Ackley(PeriodicObjectiveFunction):
    """
    Nearly flat outer region with deep hole at center (0,0).
    Has many local minima.

    Global minimum at x* = 0, f(x*) = 0.
    """

    def __init__(self, start, bounds=None):
        
        super().__init__(bounds)
        weights = torch.Tensor(start)
        self.weights = nn.Parameter(weights)
        
    def forward(self, X=None):

        if X is None:
            X = self.weights
        
        X = self.apply_period(X)
        z = - 20*torch.exp(-0.2*(torch.sqrt(0.5*torch.sum(X**2)))) - torch.exp(0.5*torch.sum(torch.cos(2*torch.pi*X))) + 20 + torch.e

        return z


# NOT DONE YET
class SummationHills(PeriodicObjectiveFunction):

    """
    Hilly function composed of a sum - allows true SGD behaviour.
    """

    def __init__(self, start, bounds=None):

        pass

        super().__init__(bounds)
        weights = torch.Tensor(start)
        self.weights = nn.Parameter(weights)

        x = 1

    def forward(self, X=None):

        if X is None:
            X = self.weights
        
        X = self.apply_period(X)
        z = 0

        return z

