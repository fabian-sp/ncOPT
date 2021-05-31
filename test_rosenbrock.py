"""
author: Fabian Schaipp
"""
import numpy as np
import matplotlib.pyplot as plt

from ncopt.sqpgs import SQP_GS
from ncopt.rosenbrock import ftest, gtest
#from ncopt.torch_obj import Net

#%%
f = ftest()
g = gtest()
#D = Net(model)

gI=[g]
gE =[]

#%%
X, Y = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
Z = np.zeros_like(X)

for j in np.arange(100):
    for i in np.arange(100):
        Z[i,j] = f.eval(np.array([X[i,j], Y[i,j]]))


plt.figure()
plt.contourf(X,Y,Z, levels = 20)

for i in range(20):
    x_k, x_hist, SP = SQP_GS(f, gI, gE, tol = 1e-8, verbose = False)
    print(x_k)
    plt.plot(x_hist[:,0], x_hist[:,1], c = "silver", lw = 1, ls = '--', alpha = 0.5)
    plt.scatter(x_k[0], x_k[1], marker = "*", s = 200, c = "silver", alpha = 1, zorder = 100)
    
plt.scatter([0.7], [0.5], marker = "*", s = 200, c = "gold", alpha = 1, zorder = 200)   

plt.xlim(-2,2)
plt.ylim(-2,2)


#%%
xsol1 = np.array([0.7071067,  0.49999994])
g.eval(xsol1)
f.eval(xsol1)




