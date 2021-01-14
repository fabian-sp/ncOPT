import numpy as np
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
    beta_kheta = 0.8
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
        D_G = list()
        for j in np.arange(nI):
            D_G.append(compute_gradients(G[j], B_G[j]))
 
        
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
        cvx_tmp3 = np.hstack((np.vstack(D_G), cvx_tmp2, np.zeros((sample_G*nI,1))))
        
        cvx_Q = np.vstack((cvx_Q, cvx_tmp3))
        
        # construct r >= 0
        r_nonneg_lhs = np.hstack((np.zeros((nI,dim)), -np.eye(nI), np.zeros((nI,1))))
        r_nonneg_rhs = np.zeros(nI)
        cvx_Q = np.vstack((cvx_Q, r_nonneg_lhs))
        
        # construct rhs
        f_k = f.eval(x_k)
        gvals_k = np.array([G[j].eval(x_k) for j in range(nI)])
        
        cvx_rhs = np.hstack((np.repeat(-f_k, sample_f), np.repeat(-gvals_k, sample_G)))
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
        g_k = lambda_f @ D_f + np.sum([lambda_G[j] @ D_G[j] for j in range(nI)], axis = 0)
        
        
        v_k = np.maximum(gvals_k, 0).sum()
        phi_k = rho*f_k + v_k
        
        
        delta_q = phi_k - q_rho(d_k, rho, H, f_k, gvals_k, D_f, D_G) 
        assert delta_q >= -1e-5
        assert np.abs(lambda_f.sum() - rho) <= 1e-6
        
        Gvals_samples = [np.array([G[j].eval(B_G[j][i,:]) for i in np.arange(sample_G)]) for j in np.arange(nI)]
        
        term3 = np.max([np.max(lambda_G[j] * Gvals_samples[j]) for j in np.arange(nI)])        
        new_E_k = np.max([np.linalg.norm(g_k, np.inf), np.max(gvals_k), term3])
        
        E_k = min(E_k, new_E_k)
        
        if verbose:
            print(out_fmt % (iter_k, f_k, np.max(gvals_k), E_k, step))
        
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
                theta *= beta_kheta
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
        