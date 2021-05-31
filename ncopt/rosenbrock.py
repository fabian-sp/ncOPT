import numpy as np
import torch

class ftest:
    
    def __init__(self, w = 8):
        self.name = 'rosenbrock'
        self.dim = 2
        self.w = w
        
    def eval(self, x):
        
        return self.w*np.abs(x[0]**2-x[1]) + (1-x[0])**2
    
    def differentiable(self, x):
        return np.abs(x[0]**2 - x[1]) > 1e-10
    
    def grad(self, x):
        a = np.array([-2+x[0], 0])
        
        sign = np.sign(x[0]**2 -x[1])
        
        if sign == 1:
            b = np.array([2*x[0], -1])
        elif sign == -1:
            b = np.array([-2*x[0], 1])
        else:
            b = np.array([-2*x[0], 1])
         
        #b = np.sign(x[0]**2 -x[1]) * np.array([2*x[0], -1])
        
        return a + b
    
class gtest:
    
    def __init__(self, c1 = np.sqrt(2), c2 = 2.):
        self.name = 'max'        
        self.c1 = c1
        self.c2 = c2
        return
    
    def eval(self, x):
        return np.maximum(self.c1*x[0], self.c2*x[1]) - 1
    
    def differentiable(self, x):
        return np.abs(self.c1*x[0] -self.c2*x[1]) > 1e-10
    
    def grad(self, x):
        
        sign = np.sign(self.c1*x[0] - self.c2*x[1])
        if sign == 1:
            g = np.array([self.c1, 0])
        elif sign == -1:
            g = np.array([0, self.c2])
        else:
            g = np.array([0, self.c2])
        return g
    

class Net:
    def __init__(self, D):
        self.name = 'pytorchNN'
        self.D = D
        
        self.D.zero_grad()
        
        self.dimIn = self.D[0].weight.shape[1]
        
        # set mode to evaluation
        self.D.train(False)
        
        if type(self.D[-1]) == torch.nn.ReLU:
            self.dimOut = self.D[-2].weight.shape[0]
        else:
            self.dimOut = self.D[-1].weight.shape[0]
 
        return
    
    def eval(self, x):      
        assert len(x) == self.dimIn, f"Input for NN has wrong dimension, required dimension is {self.dimIn}."
        
        return self.D.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    
    def grad(self, x):
        assert len(x) == self.dimIn, f"Input for NN has wrong dimension, required dimension is {self.dimIn}."
        
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)
        
        y_torch = self.D(x_torch)
        y_torch.backward()

        return x_torch.grad.data.numpy()
          