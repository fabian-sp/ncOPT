"""
author: Fabian Schaipp
"""
import numpy as np
import matplotlib.pyplot as plt

from ncopt.sqpgs import SQP_GS
from ncopt.funs import f_rosenbrock, g_max, g_upper_bound
#from ncopt.torch_obj import Net

#%%
f = f_rosenbrock()
g = g_max()
g1 = g_upper_bound(lhs=5)
#D = Net(model)

# inequality constraints (list of functions)
gI = [g,g1]
# equality constraints (list of scalar functions)
gE = []

xstar = np.array([1/np.sqrt(2), 0.5])

#%%
X, Y = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
Z = np.zeros_like(X)

for j in np.arange(100):
    for i in np.arange(100):
        Z[i,j] = f.eval(np.array([X[i,j], Y[i,j]]))


fig, ax = plt.subplots()
ax.contourf(X,Y,Z, levels = 20)
ax.scatter(xstar[0], xstar[1], marker = "*", s = 200, c = "gold", alpha = 1, zorder = 200)   


for i in range(20):
    x_k, x_hist, SP = SQP_GS(f, gI, gE, tol = 1e-4, max_iter = 50, verbose = False)
    print(x_k)
    ax.plot(x_hist[:,0], x_hist[:,1], c = "silver", lw = 0.7, ls = '--', alpha = 0.5)
    ax.scatter(x_k[0], x_k[1], marker = "+", s = 50, c = "k", alpha = 1, zorder = 210)
    
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)





