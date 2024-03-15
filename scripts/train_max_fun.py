"""
Script for training a neural network representing the function:

    f(x): x -> max(c1*x[0], c2*x[1]) - 1

Serves as example how to use neural networks as constraint functions for SQP-GS.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from ncopt.functions.max_linear import MaxOfLinear

# %% Generate data

c1 = np.sqrt(2)
c2 = 2.0


@np.vectorize
def g(x0, x1):
    return np.maximum(c1 * x0, c2 * x1) - 1


def generate_data(grid_points):
    x0 = 2 * np.random.randn(grid_points)
    x1 = 2 * np.random.randn(grid_points)
    x0.sort()
    x1.sort()
    X0, X1 = np.meshgrid(x0, x1)
    return X0, X1


grid_points = 500
X0, X1 = generate_data(grid_points)
Z = g(X0, X1)

# %% Preparations

tmp = np.stack((X0.reshape(-1), X1.reshape(-1))).T

# pytorch weights are in torch.float32, numpy data is float64
tX = torch.tensor(tmp, dtype=torch.float32)
tZ = torch.tensor(Z.reshape(-1), dtype=torch.float32)

num_samples = len(tX)  # number of training points

# %%
loss_fn = torch.nn.MSELoss(reduction="mean")
model = MaxOfLinear(input_dim=2, output_dim=2)

print(model.linear.weight.data)
print(model.linear.bias.data)

# testing
x = torch.tensor([[1.0, 4.0]])
print("True value: ", g(x[0, 0], x[0, 1]), ". Predicted value: ", model(x)[0].item())


# %% Training

lr = 1e-3
num_epochs = 10
batch_size = 25

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


def sample_batch(num_samples, b):
    S = torch.randint(high=num_samples, size=(b,))
    return S


for epoch in range(num_epochs):
    epoch_loss = 0
    for t in range(num_samples // batch_size):
        S = sample_batch(num_samples, batch_size)
        x_batch = tX[S]
        z_batch = tZ[S][:, None]  # dummy dimension to match model output

        optimizer.zero_grad()

        loss = loss_fn(model.forward(x_batch), z_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}: loss={np.mean(epoch_loss)}")
    scheduler.step()

print("Learned parameters:")
print(model.linear.weight.data)
print(model.linear.bias.data)


# %% Save checkpoint

path = "../data/checkpoints/max2d.pt"
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    path,
)


# %% Plot results

N_test = 200
X0_test, X1_test = generate_data(N_test)

tmp = np.stack((X0_test.reshape(-1), X1_test.reshape(-1))).T

# pytorch weights are in torch.float32, numpy data is float64
X_test = torch.tensor(tmp, dtype=torch.float32)

Z_test = model.forward(X_test).detach().numpy().squeeze()
Z_test_arr = Z_test.reshape(N_test, N_test)
Z_true = g(X0_test, X1_test)

print("Test mean squared error: ", np.mean((Z_test - Z_true.reshape(-1)) ** 2))


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = axs[0]
ax.contourf(X0_test, X1_test, Z_test_arr, cmap="magma", vmin=-10, vmax=10, levels=50)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Learned contours")

ax = axs[1]
ax.contourf(X0_test, X1_test, Z_true, cmap="magma", vmin=-10, vmax=10, levels=50)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("True contours")
