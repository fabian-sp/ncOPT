import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cx
import torch

from scipy.linalg import block_diag

# def gradient(self, X):
#     (N, dim) = X.shape
    
#     D = np.zeros(X.shape)
#     for i in np.arange(N):
#         D[i,:] = self.grad(X[i,:])
        
#     return D
    
def sample_points(x, eps, N):
    """
    sample N points uniformly distributed in eps-ball around x
    """
    dim = len(x)
    U = np.random.randn(N, dim)
    norm_U = np.linalg.norm(U, axis = 1)
    R = np.random.rand(N)**(1/dim)
    
    Z = eps * (R/norm_U)[:,np.newaxis] * U
    
    return x + Z

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
          
        
        

#%%

def q_rho(d, rho, H, f_k, g_k, D_f, D_gI):
    term3 = 0.5 * d@H@d
    term1 = rho* (f_k + np.max(D_f @ d))
    term2 = 0
    for j in np.arange(len(D_gI)):
        term2 += np.maximum(g_k[j] + D_gI[j] @ d, 0).max()
    
    return term1+term2+term3

def phi_rho(x, f, G, rho):
    s0 = rho*f.eval(x)
    s1 = np.sum([np.maximum(G[j].eval(x), 0) for j in range(len(G))])
    return s0 + s1

def compute_gradients(fun, X):
    """ computes gradients of function object f at all rows of array X
    """
    (N, dim) = X.shape
        
    D = np.zeros(X.shape)
    for i in np.arange(N):
        D[i,:] = fun.grad(X[i,:])
            
    return D   

def SQP_GS(f, G, tol = 1e-8, verbose = True):
    
    eps = 1e-1
    rho = 1e-1
    theta = 1e-1
    
    sample_f = 3
    sample_G = 3
    eta = 1e-8
    gamma = 0.5
    beta_eps = 0.5
    beta_rho = 0.5
    beta_theta = 0.8
    nu = 10
    xi_s = 1e3
    xi_y = 1e3
    xi_sy = 1e-6
    
    dim = f.dim
    nI = len(G)
    
    x_k = 2*np.random.randn(dim)
    iter_H = 10
    E_k = np.inf
    max_iter = 100
    
    x_hist = list()
    x_kmin1 = None; g_kmin1 = None;
    s_hist = np.zeros((dim, iter_H))
    y_hist = np.zeros((dim, iter_H))
    
    H = np.eye(dim)
    
    status = 'not optimal'; step = np.nan
    
    hdr_fmt = "%4s\t%10s\t%5s\t%5s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose:
        print(hdr_fmt % ("iter", "f(x_k)", "max(g_j(x_k))", "E_k", "step"))
    
    ##############################################
    # START OF LOOP
    ##############################################
    
    for iter_k in range(max_iter):
        
        if E_k <= tol:
            status = 'optimal'
            break
        
        ##############################################
        # SAMPLING
        ##############################################
        B_f = sample_points(x_k, eps, sample_f-1)
        B_f = np.vstack((x_k, B_f))
        
        B_G = list()
        for j in np.arange(nI):
            B_g = sample_points(x_k, eps, sample_G-1)
            B_g = np.vstack((x_k, B_g))
            B_G.append(B_g)
            
        # compute gradients
        # D_f = nabla f(x) x \in B_eps^f(x_k)
        D_f = compute_gradients(f, B_f) 
        D_gI = list()
        for j in np.arange(nI):
            D_gI.append(compute_gradients(G[j], B_G[j]))
 
        
        ##############################################
        # SUBPROBLEM
        ##############################################
        # variable has structure (d,r,z) in R^(dim+J+1)
        print("H EIGVALS", np.linalg.eigh(H)[0])
        cvx_P = block_diag(H, np.zeros((nI+1, nI+1)))
        cvx_q = np.hstack((np.zeros(dim), np.ones(nI), rho)) 
        
        # construct ineq constraints
        cvx_Q = np.hstack((D_f, np.zeros((sample_f, nI)), -np.ones((sample_f,1))))
        
        cvx_tmp2 = -np.kron(np.eye(nI), np.ones((sample_G,1)))
        cvx_tmp3 = np.hstack((np.vstack(D_gI), cvx_tmp2, np.zeros((sample_G*nI,1))))
        
        cvx_Q = np.vstack((cvx_Q, cvx_tmp3))
        
        # construct r >= 0
        r_nonneg_lhs = np.hstack((np.zeros((nI,dim)), -np.eye(nI), np.zeros((nI,1))))
        r_nonneg_rhs = np.zeros(nI)
        cvx_Q = np.vstack((cvx_Q, r_nonneg_lhs))
        
        # construct rhs
        f_k = f.eval(x_k)
        gI_k = np.array([G[j].eval(x_k) for j in range(nI)])
        
        cvx_rhs = np.hstack((np.repeat(-f_k, sample_f), np.repeat(-gI_k, sample_G)))
        cvx_rhs = np.hstack((cvx_rhs, r_nonneg_rhs))
        
        # solve QP and extract variables
        cx.solvers.options['show_progress'] = False
        qp = cx.solvers.qp(P = cx.matrix(cvx_P), q = cx.matrix(cvx_q), G = cx.matrix(cvx_Q), h = cx.matrix(cvx_rhs))
        d_k = np.array(qp['x'][:dim]).squeeze()
        r_k = np.array(qp['x'][dim:dim+nI]).squeeze()
        zz_k = np.array(qp['x'][-1]).squeeze()
        
        assert np.all(r_k >= -1e-5) 
        
        # extract dual variables = KKT multipliers
        lambda_f = np.array(qp['z'][:sample_f]).squeeze()
        lambda_G = [np.array(qp['z'][sample_f + sample_G*(j):sample_f + sample_G*(j+1)]).squeeze() for j in np.arange(nI)]
        g_k = lambda_f @ D_f + np.sum([lambda_G[j] @ D_gI[j] for j in range(nI)], axis = 0)
        
        
        v_k = np.maximum(gI_k, 0).sum()
        phi_k = rho*f_k + v_k
        
        
        delta_q = phi_k - q_rho(d_k, rho, H, f_k, gI_k, D_f, D_gI) 
        assert delta_q >= -1e-5
        assert np.abs(lambda_f.sum() - rho) <= 1e-6
        
        Gvals_samples = [np.array([G[j].eval(B_G[j][i,:]) for i in np.arange(sample_G)]) for j in np.arange(nI)]
        
        term3 = np.max([np.max(lambda_G[j] * Gvals_samples[j]) for j in np.arange(nI)])        
        new_E_k = np.max([np.linalg.norm(g_k, np.inf), np.max(gI_k), term3])
        
        E_k = min(E_k, new_E_k)
        
        if verbose:
            print(out_fmt % (iter_k, f_k, np.max(gI_k), E_k, step))
        
        ##############################################
        # STEP
        ##############################################
        step = delta_q > nu*eps**2 
        if step:
            alpha = 1.
            phi_new = phi_rho(x_k + alpha*d_k, f, G, rho)
            
            # Armijo step size rule
            while phi_new > phi_k - eta*alpha*delta_q:                
                alpha *= gamma
                phi_new = phi_rho(x_k + alpha*d_k, f, G, rho)
            
            
            # update Hessian
            if x_kmin1 is not None:
                s_k = x_k - x_kmin1
                s_hist = np.roll(s_hist, 1, axis = 1)
                s_hist[:,0] = s_k
                

                y_k = g_k - g_kmin1
                y_hist = np.roll(y_hist, 1, axis = 1)
                y_hist[:,0] = y_k
                
                
                hH = np.eye(dim)
                for l in np.arange(iter_H):
                    sl = s_hist[:,l]
                    yl = y_hist[:,l]
                    
                    cond = (np.linalg.norm(sl) <= xi_s*eps) and (np.linalg.norm(yl) <= xi_y*eps) and (np.inner(sl,yl) >= xi_sy*eps**2)
                    
                    if cond:
                        Hs = hH@sl
                        hH = hH - np.outer(Hs,Hs)/(sl @ Hs + 1e-16) + np.outer(yl,yl)/(yl @ sl + 1e-16)
                    
                assert np.all(np.abs(hH - hH.T) <= 1e-8), f"{H}"
                
                
                H = hH.copy()
                
                
            ####################################
            # ACTUAL STEP
            ###################################
            x_kmin1 = x_k.copy()
            g_kmin1 = g_k.copy()
            
            x_k = x_k + alpha*d_k
                    
        ##############################################
        # NO STEP
        ##############################################
        else:
            if v_k <= theta:
                theta *= beta_theta
            else:
                rho *= beta_rho
            
            eps *= beta_eps
        
        
        x_hist.append(x_k)
            
    
    
    ##############################################
    # END OF LOOP
    ##############################################
    x_hist = np.vstack(x_hist)
    
    if E_k > tol:
        status = 'max iterations reached'
    
    print(f"SQP GS terminate with status {status}")
    
    return x_k, x_hist
        
#%%
f = ftest()
g = gtest()
#D = Net(model)
#G=[D]

G=[g]


X, Y = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
Z = np.zeros_like(X)

for j in np.arange(100):
    for i in np.arange(100):
        Z[i,j] = f.eval(np.array([X[i,j], Y[i,j]]))


plt.figure()
plt.contourf(X,Y,Z, levels = 20)

for i in range(20):
    x_k, x_hist = SQP_GS(f, G, tol = 1e-8, verbose = True)
    print(x_k)
    plt.plot(x_hist[:,0], x_hist[:,1], c = "silver", lw = 1)

plt.xlim(-2,2)
plt.ylim(-2,2)
# x = np.zeros(2)
# eps = 1
# N = 1000
# Y = sample_points(x, eps, N)

# plt.scatter(Y[:,0], Y[:,1])

#%%
xsol1 = np.array([0.7071067,  0.49999994])
xsol2 = np.array([0.64982465, 0.42226049])
f.eval(xsol2)
g.eval(xsol2)
f.eval(xsol1)
#D.eval(xsol1)


#%%
dim = 10
nI = 3
nE = 2
p0 = 4
pI = 5*np.ones(nI, dtype = int)
pE = 2*np.ones(nE, dtype = int)

H = np.random.rand(dim,dim)
rho = 0.1

D_f = np.random.rand(p0+1, dim)
D_gI = [np.random.rand(pI[j]+1, dim) for j in range(nI)]
D_gE = [np.random.rand(pE[j]+1, dim) for j in range(nE)]

f_k = np.random.rand(1)
gI_k = np.random.rand(nI)
gE_k = np.random.rand(nE)

class Subproblem:
    def __init__(self, dim, nI, nE, p0, pI):
        
        self.dim = dim
        self.nI = nI
        self.nE = nE
        self.p0 = p0
        self.pI = pI
        
        self.P, self.q, self.inG, self.inh, self.nonnegG, self.nonnegh = initialize_subproblem(self.dim, self.nI, self.nE, self.p0, self.pI, self.pE)
        
        
    def update(self, H, rho, D_f, D_gI, D_gE):
        
        self.P, self.q, self.inG, self.inh = update_subproblem(self.dim, self.nI, self.nE, self.p0, self.pI, self.pE,\
                                                               H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k)
        return
    
    def solve(self):
        cx.solvers.options['show_progress'] = False
        
        iG = np.vstack((self.inG, self.nonnegG))
        ih = np.vstack((self.inh, self.nonnegh))
        
        qp = cx.solvers.qp(P = cx.matrix(self.P), q = cx.matrix(self.q), G = cx.matrix(iG), h = cx.matrix(ih))
        
        d = np.array(qp['x'][:self.dim]).squeeze()
        z = np.array(qp['x'][self.dim]).squeeze()
        rI = np.array(qp['x'][self.dim +1          : self.dim +1 +self.nI]).squeeze()
        rE = np.array(qp['x'][self.dim +1 +self.nI : ]).squeeze()
        
        assert len(rE) == self.nE
        
        return d,z,rI,rE
        
        
def initialize_subproblem(dim, nI, nE, p0, pI, pE):
    """
    dim : solution space dimension
    nI : number of inequality constraints
    nE : number of equality constraints
    p0 : number of sample points for f (excluding x_k itself)
    pI : array, number of sample points for inequality constraint (excluding x_k itself)
    pE : array, number of sample points for equality constraint (excluding x_k itself)
    """
    
    dimQP = dim+1+nI+nE
    
    P = np.zeros((dimQP,dimQP))
    q = np.zeros(dimQP)
    
    inG = np.zeros((1+p0+np.sum(1+pI)+2*np.sum(1+pE),dimQP))
    inh = np.zeros(1+p0+np.sum(1+pI)+2*np.sum(1+pE))
    
    # structure of inG (p0+1, sum(1+pI), sum(1+pE), sum(1+pE))
    inG[:p0+1,dim] = -1
    
    for j in range(nI):
        inG[p0+1+(1+pI)[:j].sum()                           :  p0+1+(1+pI)[:j].sum()                           + pI[j]+1, dim+1+j]    = -1
        
        
    for j in range(nE):
        inG[p0+1+(1+pI).sum()+(1+pE)[:j].sum()              :  p0+1+(1+pI).sum()+(1+pE)[:j].sum()              + pE[j]+1, dim+1+nI+j] = -1
        inG[p0+1+(1+pI).sum()+(1+pE).sum()+(1+pE)[:j].sum() :  p0+1+(1+pI).sum()+(1+pE).sum()+(1+pE)[:j].sum() + pE[j]+1, dim+1+nI+j] = -1
        
    
    # we have nI+nE r-variables
    nonnegG = np.hstack((np.zeros((nI+nE,dim+1)), -np.eye(nI+nE)))
    nonnegh = np.zeros(nI+nE)
 
    return P,q,inG,inh,nonnegG,nonnegh

P,q,inG,inh,nonnegG,nonnegh = initialize_subproblem(dim, nI, nE, p0, pI, pE)

def update_subproblem(dim, nI, nE, p0, pI, pE, H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k):
    
    P[:dim, :dim] = H
    q = np.hstack((np.zeros(dim), rho, np.ones(nI), np.ones(nE))) 
    
    
    inG[:p0+1, :dim] = D_f
    inh[:p0+1] = -f_k
    
    for j in range(nI):
        inG[p0+1+(1+pI)[:j].sum()                           :  p0+1+(1+pI)[:j].sum()                           + pI[j]+1, :dim]    = D_gI[j]
        inh[p0+1+(1+pI)[:j].sum()                           :  p0+1+(1+pI)[:j].sum()                           + pI[j]+1]          = -gI_k[j] 
        
    for j in range(nE):
        inG[p0+1+(1+pI).sum()+(1+pE)[:j].sum()              :  p0+1+(1+pI).sum()+(1+pE)[:j].sum()              + pE[j]+1, :dim] = D_gE[j]
        inG[p0+1+(1+pI).sum()+(1+pE).sum()+(1+pE)[:j].sum() :  p0+1+(1+pI).sum()+(1+pE).sum()+(1+pE)[:j].sum() + pE[j]+1, :dim] = -D_gE[j]
        
        inh[p0+1+(1+pI).sum()+(1+pE)[:j].sum()              :  p0+1+(1+pI).sum()+(1+pE)[:j].sum()              + pE[j]+1] = -gE_k[j]
        inh[p0+1+(1+pI).sum()+(1+pE).sum()+(1+pE)[:j].sum() :  p0+1+(1+pI).sum()+(1+pE).sum()+(1+pE)[:j].sum() + pE[j]+1] = gE_k[j]
        
   
    return P, q, inG, inh


    
    