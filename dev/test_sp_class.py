# test Subproblem class


#%%
# dim = 10
# nI = 3
# nE = 2
# p0 = 4
# pI = 5*np.ones(nI, dtype = int)
# pE = 2*np.ones(nE, dtype = int)

# H = np.eye(dim)
# rho = 0.1

# D_f = np.random.rand(p0+1, dim)
# D_gI = [np.random.rand(pI[j]+1, dim) for j in range(nI)]
# D_gE = [np.random.rand(pE[j]+1, dim) for j in range(nE)]

# f_k = np.random.rand(1)
# gI_k = np.random.rand(nI)
# gE_k = np.random.rand(nE)


# SP = Subproblem(dim, nI, nE, p0, pI, pE)
# SP.update(H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k)

# inG = SP.inG


# SP.solve()    

# d_k=SP.d

