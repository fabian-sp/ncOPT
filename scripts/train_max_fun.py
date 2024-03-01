"""
This is as rather experimental script for training a NN representing the function:

x \mapsto max(c1*x[0], c2*x[1]) - 1

The idea is to use a neural network as a constraint for SQP-GS.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR

c1 = np.sqrt(2); c2 = 2.

@np.vectorize
def g(x0,x1):
    return np.maximum(c1*x0, c2*x1) - 1

def generate_data(N):
    x0 = 2*np.random.randn(N)# * 10 - 5
    x1 = 2*np.random.randn(N)# * 10 - 5
    x0.sort();x1.sort()
    X0,X1 = np.meshgrid(x0,x1)
    return X0,X1

X0, X1 = generate_data(200)
Z = g(X0,X1)

#%%

tmp = np.stack((X0.reshape(-1),X1.reshape(-1))).T

# pytorch weights are in torch.float32, numpy data is float64!
tX = torch.tensor(tmp, dtype = torch.float32)
tZ = torch.tensor(Z.reshape(-1), dtype = torch.float32)

N = len(tX)

#%%
# # D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.

# D_in, H, D_out = 2, 200, 1

# # define model and loss function.
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )

# loss_fn = torch.nn.MSELoss(reduction='mean')

#%%

class myNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(2, 2) # layer 1
        #self.l2 = torch.nn.Linear(20, 2) # layer 2
        #self.relu = torch.nn.ReLU()
        self.max = torch.max
    def forward(self, x):
        x = self.l1(x)
        x,_ = self.max(x, dim = -1)
        return x

loss_fn = torch.nn.MSELoss(reduction='mean')

model = myNN()

# set weights manually
#model.state_dict()["l1.weight"][:] = torch.diag(torch.tensor([c1,c2]))
#model.state_dict()["l1.bias"][:] = -torch.ones(2)

print(model.l1.weight)
print(model.l1.bias)

#testing
x = torch.tensor([1.,4.])
model(x)
g(x[0], x[1])

#%%
learning_rate = 1e-3
N_EPOCHS = 11
b = 15

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, nesterov=True)

scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

def sample_batch(N, b):    
    S = torch.randint(high = N, size = (b,))
    return S

for epoch in range(N_EPOCHS):
    print(f"..................EPOCH {epoch}..................")
      
    for t in range(int(N/b)):
        
        S = sample_batch(N, b)
        x_batch = tX[S]; z_batch = tZ[S]
                
        # forward pass
        y_pred = model.forward(x_batch)
    
        # compute loss.
        loss = loss_fn(y_pred.squeeze(), z_batch)
               
        # zero gradients
        optimizer.zero_grad()
    
        # backward pass
        loss.backward()
    
        # iteration
        optimizer.step()
        
    print(model.l1.weight)
    print(model.l1.bias)

    print(loss.item())
    scheduler.step()
    #print(optimizer)
    
optimizer.zero_grad()       
    
#%% plot results

N_test = 200
X0_test,X1_test = generate_data(N_test)

tmp = np.stack((X0_test.reshape(-1), X1_test.reshape(-1))).T

# pytorch weights are in torch.float32, numpy data is float64!
X_test = torch.tensor(tmp, dtype = torch.float32)

Z_test = model.forward(X_test).detach().numpy().squeeze()

Z_test_arr = Z_test.reshape(N_test, N_test)

Z_true = g(X0_test,X1_test).reshape(-1)


fig, axs = plt.subplots(1,2)
axs[0].scatter(tmp[:,0], tmp[:,1], c = Z_test)
#axs[1].scatter(tmp[:,0], tmp[:,1], c = Z_true)
axs[1].scatter(tmp[:,0], tmp[:,1], c = Z_test-Z_true, vmin = -1e-1, vmax = 1e-1, cmap = "coolwarm")


#%%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
ax.plot_surface(X0_test, X1_test, Z_test_arr, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

#%% test auto-diff gradient

x0 = torch.tensor([np.sqrt(2),0.5], dtype = torch.float32)

x0.requires_grad_(True)
model.zero_grad()

y0 = model(x0)
y0.backward()

x0.grad.data

W = model[-3].weight.detach().numpy()



