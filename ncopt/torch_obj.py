"""
author: Fabian Schaipp
"""
import numpy as np
import torch

class Net:
    def __init__(self, D, dimOut = None):
        self.name = 'pytorch_Net'
        self.D = D
        
        self.D.zero_grad()
        
        self.dimIn = self.D[0].weight.shape[1]
        
        # set mode to evaluation
        self.D.train(False)
        
        #if type(self.D[-1]) == torch.nn.ReLU:
        if dimOut is None:
            print("Caution: output dimension of Net is not specified and derived from last module!")
            self.dimOut = self.D[-1].weight.shape[0]
        else:
            self.dimOut = dimOut
        return
    
    def eval(self, x):      
        assert len(x) == self.dimIn, f"Input for Net has wrong dimension, required dimension is {self.dimIn}."
        
        return self.D.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    
    def grad(self, x):
        assert len(x) == self.dimIn, f"Input for Net has wrong dimension, required dimension is {self.dimIn}."
        
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)
        
        y_torch = self.D(x_torch)
        y_torch.backward()

        return x_torch.grad.data.numpy()
          