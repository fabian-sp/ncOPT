"""
@author: Fabian Schaipp

Implements the SQP-GS algorithm from 

    Frank E. Curtis and Michael L. Overton, A sequential quadratic programming algorithm for nonconvex, nonsmooth constrained optimization, 
    SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

The notation of the code tries to follow closely the notation of the paper.
"""

import numpy as np
import cvxopt as cx

import copy
import time
from typing import Optional

from .defaults import DEFAULT_ARG, DEFAULT_OPTION

class SQPGS:
    def __init__(self,
                 f,
                 gI,
                 gE,
                 x0: Optional[np.array]=None,
                 tol: float=DEFAULT_ARG.tol, 
                 max_iter: int=DEFAULT_ARG.max_iter, 
                 verbose: bool=DEFAULT_ARG.verbose, 
                 options: dict={},
                 assert_tol: float=DEFAULT_ARG.assert_tol
                 ) -> None:
        
        if tol < 0:
            raise ValueError(f"Tolerance must be non-negative, but was specified as {tol}.")
        if max_iter < 0:
            raise ValueError(f"Maximum number of iterations must be non-negative, but was specified as {max_iter}.")
        if assert_tol < 0:
            raise ValueError(f"Assertion tolerance must be non-negative, but was specified as {assert_tol}.")
        
        self.f = f
        self.gI = gI
        self.gE = gE
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.options = options
        self.assert_tol = assert_tol

        # Set options/hyperparameters
        # (defaults are chose according to recommendations in paper)
        self.options = copy.deepcopy(DEFAULT_OPTION.copy())
        self.options.update(options)

        ###############################################################
        ########## Extract dimensions
        
        # extract dimensions of constraints
        self.dim = self.f.dim
        self.dimI = np.array([g.dimOut for g in self.gI], dtype=int)
        self.dimE = np.array([g.dimOut for g in self.gE], dtype=int)
        
        
        self.nI_ = len(self.gI) # number of inequality function objects
        self.nE_ = len(self.gE) # number of equality function objects
        
        self.nI = sum(self.dimI) # number of inequality costraints 
        self.nE = sum(self.dimE) # number of equality costraints
        
        ###############################################################
        ########## Initialize
        
        self.status = 'not optimal'
        
        # starting point
        if x0 is None:
            self.x_k = np.zeros(self.dim)
        else:
            self.x_k = x0.copy()
            
        return
    
    def solve(self):
        ###############################################################
        ########## Set all hyperparameters 

        eps = self.options['eps'] # sampling radius
        rho = self.options['rho']
        theta = self.options['theta']
          
        eta = self.options['eta']
        gamma = self.options['gamma']
        beta_eps = self.options['beta_eps']
        beta_rho = self.options['beta_rho']
        beta_theta = self.options['beta_theta']
        nu = self.options['nu']
        xi_s = self.options['xi_s']
        xi_y = self.options['xi_y']
        xi_sy = self.options['xi_sy']
        iter_H = self.options['iter_H']

        p0 = self.options['num_points_obj']                                  # sample points for objective
        pI_ = self.options['num_points_gI'] * np.ones(self.nI_, dtype=int)   # sample points for ineq constraint
        pE_ = self.options['num_points_gE'] * np.ones(self.nE_, dtype=int)   # sample points for eq constraint
        
        pI = np.repeat(pI_, self.dimI)
        pE = np.repeat(pE_, self.dimE)
        ###############################################################

        self.SP = SubproblemSQPGS(self.dim, p0, pI, pE, self.assert_tol)
        
        E_k = np.inf               # for stopping criterion
        x_hist = [self.x_k]
        x_kmin1 = None             # last iterate
        g_kmin1 = None             # 

        # Hessian matrix
        H = np.eye(self.dim)
        s_hist = np.zeros((self.dim, iter_H))
        y_hist = np.zeros((self.dim, iter_H))
             
        do_step = False

        hdr_fmt = "%4s\t%10s\t%5s\t%5s\t%10s\t%10s"
        out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10s"
        if self.verbose:
            print(hdr_fmt % ("iter", "f(x_k)", "max(g_j(x_k))", "E_k", "step", "subproblem status"))
        
        self.timings = {'total': [], 'sp_update': [], 'sp_solve': []}
        ##############################################
        # START OF LOOP
        ##############################################
        for iter_k in range(self.max_iter):
            
            t0 = time.perf_counter()
            if E_k <= self.tol:
                self.status = 'optimal'
                break
            
            ##############################################
            # SAMPLING
            ##############################################
            B_f = sample_points(self.x_k, eps, p0)
            B_f = np.vstack((self.x_k, B_f))
            
            B_gI = list()
            for j in np.arange(self.nI_):
                B_j = sample_points(self.x_k, eps, pI_[j])
                B_j = np.vstack((self.x_k, B_j))
                B_gI.append(B_j)
                
            B_gE = list()
            for j in np.arange(self.nE_):
                B_j = sample_points(self.x_k, eps, pE_[j])
                B_j = np.vstack((self.x_k, B_j))
                B_gE.append(B_j)
                
                
            ####################################
            # COMPUTE GRADIENTS AND EVALUATE
            ###################################
            D_f = compute_gradients(self.f, B_f)[0] # returns list, always has one element
            
            D_gI = list()
            for j in np.arange(self.nI_):
                D_gI += compute_gradients(self.gI[j], B_gI[j])
                
            D_gE = list()
            for j in np.arange(self.nE_):
                D_gE += compute_gradients(self.gE[j], B_gE[j])
    
            f_k = self.f.eval(self.x_k)
            # hstack cannot handle empty lists!
            if self.nI_ > 0:
                gI_k = np.hstack([self.gI[j].eval(self.x_k) for j in range(self.nI_)])
            else:
                gI_k = np.array([])
            
            if self.nE_ > 0:
                gE_k = np.hstack([self.gE[j].eval(self.x_k) for j in range(self.nE_)])
            else:
                gE_k = np.array([])
            
            ##############################################
            # SUBPROBLEM
            ##############################################
            
            t01 = time.perf_counter()
            self.SP.update(H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k)
            t11 = time.perf_counter()
            self.SP.solve()
            t21 = time.perf_counter()

            self.timings['sp_update'].append(t11-t01)
            self.timings['sp_solve'].append(t21-t11)

            d_k = self.SP.d.copy()
            # compute g_k from paper 
            g_k = self.SP.lambda_f @ D_f \
                    + np.sum([self.SP.lambda_gI[j] @ D_gI[j] for j in range(self.nI)], axis = 0)  \
                    + np.sum([self.SP.lambda_gE[j] @ D_gE[j] for j in range(self.nE)], axis = 0)
                                    
            # evaluate v(x) at x=x_k
            v_k = np.maximum(gI_k, 0).sum() + np.sum(np.abs(gE_k))
            phi_k = rho*f_k + v_k  
            delta_q = phi_k - q_rho(d_k, rho, H, f_k, gI_k, gE_k, D_f, D_gI, D_gE) 
            
            assert delta_q >= -self.assert_tol, f"Value is supposed to be non-negative, but is {delta_q}."
            assert np.abs(self.SP.lambda_f.sum() - rho) <= self.assert_tol, f"Value is supposed to be negative, but is {np.abs(self.SP.lambda_f.sum() - rho)}."
                     
            if self.verbose:
                print(out_fmt % (iter_k, f_k, np.max(np.hstack((gI_k,gE_k))), E_k, do_step, self.SP.status))
            
            new_E_k = stop_criterion(self.gI, self.gE, g_k, self.SP, gI_k, gE_k, B_gI, B_gE, self.nI_, self.nE_, pI, pE)
            E_k = min(E_k, new_E_k)
            
            ##############################################
            # STEP
            ##############################################
            
            do_step = delta_q > nu*eps**2    # Flag whether step is taken or not
            if do_step:
                alpha = 1.
                phi_new = phi_rho(self.x_k + alpha*d_k, self.f, self.gI, self.gE, rho)
                
                # Armijo step size rule
                while phi_new > phi_k - eta*alpha*delta_q:                
                    alpha *= gamma
                    phi_new = phi_rho(self.x_k + alpha*d_k, self.f, self.gI, self.gE, rho)
                    
                # update Hessian
                if x_kmin1 is not None:
                    s_k = self.x_k - x_kmin1
                    s_hist = np.roll(s_hist, 1, axis=1)
                    s_hist[:,0] = s_k
                    
                    y_k = g_k - g_kmin1
                    y_hist = np.roll(y_hist, 1, axis=1)
                    y_hist[:,0] = y_k
                                    
                    hH = np.eye(self.dim)
                    for l in np.arange(iter_H):
                        sl = s_hist[:,l]
                        yl = y_hist[:,l]
                        
                        cond1 = (np.linalg.norm(sl) <= xi_s*eps) and (np.linalg.norm(yl) <= xi_y*eps)
                        cond2 = (np.inner(sl,yl) >= xi_sy*eps**2)
                        cond = cond1 and cond2 
                        
                        if cond:
                            Hs = hH @ sl
                            hH = hH - np.outer(Hs,Hs)/(sl @ Hs + 1e-16) + np.outer(yl,yl)/(yl @ sl + 1e-16)

                    H = hH.copy()
                    
                ####################################
                # ACTUAL STEP
                ###################################
                x_kmin1 = self.x_k.copy()
                g_kmin1 = g_k.copy()
                
                self.x_k = self.x_k + alpha*d_k
                        
            ##############################################
            # NO STEP
            ##############################################
            else:
                if v_k <= theta:
                    theta *= beta_theta
                else:
                    rho *= beta_rho
                
                eps *= beta_eps
            
            
            x_hist.append(self.x_k)
            t1 = time.perf_counter()
            self.timings['total'].append(t1-t0)
            
        ##############################################
        # END OF LOOP
        ##############################################
        self.x_hist = np.vstack(x_hist)
    
        if E_k > self.tol:
            self.status = 'max iterations reached'

        print(f"SQP-GS has terminated with status: {self.status}.")

        return self.x_k

def sample_points(x, eps, N):
    """
    sample N points uniformly distributed in eps-ball around x
    """
    dim = len(x)
    U = np.random.randn(N, dim)
    norm_U = np.linalg.norm(U, axis=1)
    R = np.random.rand(N)**(1/dim)    
    Z = eps * (R/norm_U)[:,np.newaxis] * U
    
    return x + Z


def q_rho(d, rho, H, f_k, gI_k, gE_k, D_f, D_gI, D_gE):
    term1 = rho* (f_k + np.max(D_f @ d))
    
    term2 = 0
    for j in np.arange(len(D_gI)):
        term2 += np.maximum(gI_k[j] + D_gI[j] @ d, 0).max()
    
    term3 = 0
    for l in np.arange(len(D_gE)):
        term3 += np.abs(gE_k[l] + D_gE[l] @ d).max()
    
    term4 = 0.5 * d.T@H@d
    return term1+term2+term3+term4

def phi_rho(x, f, gI, gE, rho):
    term1 = rho*f.eval(x)
    
    # inequalities
    if len(gI) > 0:
        term2 = np.sum(np.hstack([np.maximum(gI[j].eval(x), 0) for j in range(len(gI))]))
    else:
        term2 = 0
    # equalities: max(x,0) + max(-x,0) = abs(x)
    if len(gE) > 0:
        term3 = np.sum(np.hstack([np.abs(gE[l].eval(x)) for l in range(len(gE))]))
    else:
        term3 = 0
        
    return term1+term2+term3

def stop_criterion(gI, gE, g_k, SP, gI_k, gE_k, B_gI, B_gE, nI_, nE_, pI, pE):
    """
    computes E_k in the paper
    """
    val1 = np.linalg.norm(g_k, np.inf)
    
    # as gI or gE could be empty, we need a max value for empty arrays --> initial argument
    val2 = np.max(gI_k, initial=-np.inf)
    val3 = np.max(np.abs(gE_k), initial=-np.inf)
    
    gI_vals = list()
    for j in np.arange(nI_):
        gI_vals += eval_ineq(gI[j], B_gI[j])
    
    val4 = -np.inf
    for j in np.arange(len(gI_vals)):
        val4 = np.maximum(val4, np.max(SP.lambda_gI[j] * gI_vals[j]))
    
    gE_vals = list()
    for j in np.arange(nE_):
        gE_vals += eval_ineq(gE[j], B_gE[j])
    
    val5 = -np.inf
    for j in np.arange(len(gE_vals)):
        val5 = np.maximum(val5, np.max(SP.lambda_gE[j] * gE_vals[j]))
    
    return np.max(np.array([val1, val2, val3, val4, val5]))

def eval_ineq(fun, X):
    """
    evaluate function at multiple inputs
    needed in stop_criterion
    
    Returns
    -------
    list of array, number of entries = fun.dimOut 
    """
    (N, _) = X.shape
    D = np.zeros((N, fun.dimOut))
    for i in np.arange(N):
        D[i,:,] = fun.eval(X[i,:])
    
    return [D[:,j] for j in range(fun.dimOut)]


def compute_gradients(fun, X):
    """ 
    computes gradients of function object f at all rows of array X
    
    Returns
    -------
    list of 2d-matrices, length of fun.dimOut
    """
    (N, dim) = X.shape
    
    # fun.grad returns Jacobian, i.e. dimOut x dim
    D = np.zeros((N, fun.dimOut, dim))
    for i in np.arange(N):
        D[i,:,:] = fun.grad(X[i,:])
    
    return [D[:,j,:] for j in np.arange(fun.dimOut)]   
#%%

class SubproblemSQPGS:
    def __init__(self, dim, p0, pI, pE, assert_tol):
        """
        dim : solution space dimension
        p0 : number of sample points for f (excluding x_k itself)
        pI : array, number of sample points for inequality constraint (excluding x_k itself)
        pE : array, number of sample points for equality constraint (excluding x_k itself)
        """
        
        self.dim = dim
        self.nI = len(pI)
        self.nE = len(pE)
        self.p0 = p0
        self.pI = pI
        self.pE = pE
        
        self.assert_tol = assert_tol

        self.P, self.q, self.inG, self.inh, self.nonnegG, self.nonnegh = self.initialize()
        
    
    def solve(self):
        """
        This solves the quadratic program. In every iteration, you should call self.update() before solving in order to have the correct subproblem data.
        
        self.d: array
            search direction
            
        self.lambda_f: array
            KKT multipier for objective.
            
        self.lambda_gE: list
            KKT multipier for equality constraints. 
        
        self.lambda_gI: list
            KKT multipier for inequality constraints.    

        """
        cx.solvers.options['show_progress'] = False
        
        iG = np.vstack((self.inG, self.nonnegG))
        ih = np.hstack((self.inh, self.nonnegh))
        
        qp = cx.solvers.qp(P=cx.matrix(self.P), q=cx.matrix(self.q), G=cx.matrix(iG), h=cx.matrix(ih))
        
        self.status = qp["status"]
        self.cvx_sol_x = np.array(qp['x']).squeeze()
        
        self.d = self.cvx_sol_x[:self.dim]
        self.z = self.cvx_sol_x[self.dim]

        self.rI = self.cvx_sol_x[self.dim +1           : self.dim +1 +self.nI]
        self.rE = self.cvx_sol_x[self.dim +1 + self.nI : ]
        
        assert len(self.rE) == self.nE
        assert np.all(self.rI >= -self.assert_tol) , f"Array should be non-negative, but minimal value is {np.min(self.rI)}."
        assert np.all(self.rE >= -self.assert_tol), f"Array should be non-negative, but minimal value is {np.min(self.rE)}."
        
        # extract dual variables = KKT multipliers
        self.cvx_sol_z = np.array(qp['z']).squeeze()
        lambda_f = self.cvx_sol_z[:self.p0+1]
        
        lambda_gI = list()
        for j in np.arange(self.nI):
            start_ix = self.p0+1+(1+self.pI)[:j].sum()
            lambda_gI.append( self.cvx_sol_z[start_ix : start_ix + 1+self.pI[j]]  )
        
        lambda_gE = list()
        for j in np.arange(self.nE):
            start_ix = self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()
            
            # from ineq with +
            vec1 = self.cvx_sol_z[start_ix : start_ix + 1+self.pE[j]]
            
            # from ineq with -
            vec2 = self.cvx_sol_z[start_ix+(1+self.pE).sum() : start_ix + (1+self.pE).sum() + 1+self.pE[j]]
            
            # see Direction.m line 620 in the original Matlab code
            lambda_gE.append(vec1-vec2)
     
        self.lambda_f = lambda_f.copy()
        self.lambda_gI = lambda_gI.copy()
        self.lambda_gE = lambda_gE.copy()
        
        return 
        
        
    def initialize(self):
        """
        The quadratic subrpoblem we solve in every iteration is of the form:
        
        min_y 1/2* yPy + q*y subject to Gy <= h
        
        variable structure: y=(d,z,rI,rE) with
        d = search direction
        z = helper variable for objective
        rI = helper variable for inequality constraints
        rI = helper variable for equality constraints
        
        This function initializes the variables P,q,G,h. The entries which change in every iteration are then updated in self.update()
        
        G and h consist of two parts:
            1) inG, inh: the inequalities from the paper
            2) nonnegG, nonnegh: nonnegativity bounds rI >= 0, rE >= 0
        """
        
        dimQP = self.dim+1 + self.nI + self.nE
        
        P = np.zeros((dimQP, dimQP))
        q = np.zeros(dimQP)
        
        inG = np.zeros((1 + self.p0+np.sum(1+self.pI) + 2*np.sum(1+self.pE), dimQP))
        inh = np.zeros( 1 + self.p0+np.sum(1+self.pI) + 2*np.sum(1+self.pE))
        
        # structure of inG (p0+1, sum(1+pI), sum(1+pE), sum(1+pE))
        inG[:self.p0+1, self.dim] = -1
        
        for j in range(self.nI):
            inG[self.p0+1+(1+self.pI)[:j].sum()                                     :  self.p0+1+(1+self.pI)[:j].sum()                                      + self.pI[j]+1, self.dim+1+j]         = -1
            
        for j in range(self.nE):
            inG[self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()                   :  self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()                    + self.pE[j]+1, self.dim+1+self.nI+j] = -1
            inG[self.p0+1+(1+self.pI).sum()+(1+self.pE).sum()+(1+self.pE)[:j].sum() :  self.p0+1+(1+self.pI).sum()+(1+self.pE).sum()+(1+self.pE)[:j].sum()  + self.pE[j]+1, self.dim+1+self.nI+j] = -1
            
        # we have nI+nE r-variables
        nonnegG = np.hstack((np.zeros((self.nI + self.nE, self.dim + 1)), -np.eye(self.nI + self.nE)))
        nonnegh = np.zeros(self.nI + self.nE)
     
        return P,q,inG,inh,nonnegG,nonnegh


    def update(self, H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k):
        """

        Parameters
        ----------
        H : array
            Hessian approximation
        rho : float
            parameter
        D_f : array
            gradient of f at the sampled points
        D_gI : list
            j-th element is the gradient array of c^j at the sampled points.
        D_gE : list
            j-th element is the gradient array of h^j at the sampled points.
        f_k : float
            evaluation of f at x_k.
        gI_k : array
            evaluation of inequality constraints at x_k.
        gE_k : array
            evaluation of equality constraints at x_k.
        
        Returns
        -------
        None.

        """
        self.P[:self.dim, :self.dim] = H
        self.q = np.hstack((np.zeros(self.dim), rho, np.ones(self.nI), np.ones(self.nE))) 
        
        self.inG[:self.p0+1, :self.dim] =  D_f
        self.inh[:self.p0+1]            = -f_k
        
        for j in range(self.nI):
            self.inG[self.p0+1+(1+self.pI)[:j].sum()        :  self.p0+1+(1+self.pI)[:j].sum()        + self.pI[j]+1, :self.dim]    =  D_gI[j]
            self.inh[self.p0+1+(1+self.pI)[:j].sum()        :  self.p0+1+(1+self.pI)[:j].sum()        + self.pI[j]+1]               = -gI_k[j] 
            
        for j in range(self.nE):
            self.inG[self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()                   :  self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()                   + self.pE[j]+1, :self.dim]  =  D_gE[j]
            self.inG[self.p0+1+(1+self.pI).sum()+(1+self.pE).sum()+(1+self.pE)[:j].sum() :  self.p0+1+(1+self.pI).sum()+(1+self.pE).sum()+(1+self.pE)[:j].sum() + self.pE[j]+1, :self.dim]  = -D_gE[j]
            
            self.inh[self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()                   :  self.p0+1+(1+self.pI).sum()+(1+self.pE)[:j].sum()                   + self.pE[j]+1]             = -gE_k[j]
            self.inh[self.p0+1+(1+self.pI).sum()+(1+self.pE).sum()+(1+self.pE)[:j].sum() :  self.p0+1+(1+self.pI).sum()+(1+self.pE).sum()+(1+self.pE)[:j].sum() + self.pE[j]+1]             =  gE_k[j]
            
       
        return        
    


