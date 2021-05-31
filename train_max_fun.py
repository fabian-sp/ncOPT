"""
Script for training a NN representing the function

x \mapsto max(c1*x[0], c2*x[1]) - 1

The net can be used as a constraint for SQP GS
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

    x0 = np.random.rand(N) * 10 - 5
    x1 = np.random.rand(N) * 10 - 5
    x0.sort();x1.sort()
    X0,X1 = np.meshgrid(x0,x1)
    
    return X0,X1

X0,X1 = generate_data(200)
Z = g(X0,X1)


#%%
# b is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
b, D_in, H, D_out = 10, 2, 200, 1

tmp = np.stack((X0.reshape(-1),X1.reshape(-1))).T

# pytorch weights are in torch.float32, numpy data is float64!
x = torch.tensor(tmp, dtype = torch.float32)
z = torch.tensor(Z.reshape(-1), dtype = torch.float32)

N = len(x)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='mean')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, nesterov=True)

scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

def sample_batch(N, b):
    
    S = torch.randint(high = N, size = (b,))
    return S

N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    print(f"..................EPOCH {epoch}..................")
    
    
    for t in range(int(N/b)):
        
        S = sample_batch(N, b)
        x_batch = x[S]; z_batch = z[S]
        
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model.forward(x_batch)
    
    
        # Compute and print loss.
        loss = loss_fn(y_pred.squeeze(), z_batch)
               
        # zero gradients
        optimizer.zero_grad()
    
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()
    
        # iteration
        optimizer.step()

    print(loss.item())
    scheduler.step()
    print(optimizer)
    
 

optimizer.zero_grad()       
    
#%% plot results

N_test = 200
X0_test,X1_test = generate_data(N_test)

tmp = np.stack((X0_test.reshape(-1),X1_test.reshape(-1))).T

# pytorch weights are in torch.float32, numpy data is float64!
X_test = torch.tensor(tmp, dtype = torch.float32)

Z_test = model.forward(X_test).detach().numpy().squeeze()

Z_test_arr = Z_test.reshape(N_test, N_test)

Z_true = g(X0_test,X1_test).reshape(-1)


fig, axs = plt.subplots(1,2)
axs[0].scatter(tmp[:,0], tmp[:,1], c = Z_test)
#axs[1].scatter(tmp[:,0], tmp[:,1], c = Z_true)
axs[1].scatter(tmp[:,0], tmp[:,1], c = Z_test-Z_true, vmin = -1e-1, vmax = 1e-1, cmap = "coolwarm")



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

W = model[-1].weight.detach().numpy()

#%%
# G_test = torch.zeros((X_test.shape))

# for j in range(len(X_test)):
#     x0 = X_test[j]
#     x0.requires_grad_(True)
    
#     #model.zero_grad()

#     y0 = model(x0)
#     y0.backward()

#     G_test[j] = x0.grad.data

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Plot the surface.
# G_test_arr = np.linalg.norm(G_test.numpy(), axis = 1).reshape(N_test, N_test)
# ax.plot_surface(X0_test, X1_test, G_test_arr, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)





