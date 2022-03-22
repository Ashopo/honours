import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim.optimizer import Optimizer
import itertools
import plotly.graph_objects as go
from typing import Tuple

class OptimisationProblem():

    def __init__(
        self, 
        func: nn.Module, 
        opt: Optimizer,
        n_epochs: int
    ):
        self.func = func
        self.opt = opt
        self.n_epochs = n_epochs

    def run(self):
        func = self.func
        opt = self.opt
        losses, params, preds = [], [], []
        params.append(list(func.parameters())[0].data.detach().clone().numpy())

        for i in range(self.n_epochs):
            pred = func()
            loss = pred
            #print("somethings gotta change here for later - neural nets")
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(float(loss))
            params.append(list(func.parameters())[0].data.detach().clone().numpy())
            preds.append(float(pred))
        
        params = [list(_) for _ in list(zip(*params))]

        self.losses = losses
        self.params = params
        self.preds = preds

        return losses, params, preds
    
    def visualise(
        self, 
        xlim: Tuple,
        ylim: Tuple,
        res: float,
        render: str = None
    ):
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(self.losses)), y=self.losses))
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

        elif render == "contour":
            X = np.arange(xlim[0], xlim[1], res)
            Y = np.arange(ylim[0], ylim[1], res)
            grid = [np.array(_) for _ in itertools.product(X,Y)]
            Z = [float(self.func.forward(torch.Tensor(_))) for _ in grid]
            Z = np.array(Z).reshape(len(X), len(Y)).T

            fig2 = go.Figure()
            fig2.add_trace(go.Contour(x=X, y=Y, z=Z))
            fig2.add_trace(go.Scatter(x=self.params[0], 
                                      y=self.params[1],
                                      text=list(zip(self.preds, list(np.arange(0, len(self.preds)))))
                                    )
            )

        else:
            raise ValueError("""
                            Required render option does not exist. Allowed renders include:
                                        - 3d
                                        - contour
                             """)

        fig2.update_layout(
            autosize=False,
            width=800,
            height=800
        )
        
        return fig1, fig2
    

class GaussianWellModel(nn.Module):

    def __init__(self):
        
        super().__init__()
        weights = torch.Tensor([1,-18])
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

        z = 0
        for c, d, s in zip(self.centers, self.depths, self.sigmas):
            v = X - c
            z += d/(2*np.pi*s**2) * torch.exp(-torch.dot(v,v.T)/(2*s**2))

        return z

class AlpineN1(nn.Module):

    def __init__(self, start):
        
        super().__init__()
        weights = torch.Tensor(start)
        self.weights = nn.Parameter(weights)
        
    def forward(self, X=None):
        """
        Implement function to be optimised.
        """

        if X is None:
            X = self.weights

        z = torch.abs(X*torch.sin(X) + 0.1*X)
        return torch.sum(z)
