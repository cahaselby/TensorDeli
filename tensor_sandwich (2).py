
"""
27 MAR 2023

This is the module needed to run tensor sandwich trial

author: Cullen Haselby 
"""
#########################
# IMPORTS
#########################

import numpy as np
from scipy.linalg import khatri_rao,hilbert,null_space,subspace_angles, hadamard, qr
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac, tucker
import cvxpy
from solver import Solver
from sklearn.utils import check_array
import timeit

#########################
# Helper functions
#########################

def add_noise(T,relnoise):

    Noise = np.random.normal(size=T.shape)
    #Normalize 
    Noise = Noise / tl.norm(Noise)
    #Scale relative to norm of T
    Noise = (relnoise *tl.norm(T) )* Noise

    return T +  Noise

def rel_error(approxT,exactT):
    return tl.norm(approxT - exactT) / tl.norm(exactT)

def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.lstsq(A[M], B[M],rcond=None)[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        output = np.squeeze(np.linalg.solve(T, rhs)).T # transpose to get r x n
    except np.linalg.LinAlgError:
        print("system is probably singular, switching to least square mode")
        
        output = np.empty((A.shape[1], B.shape[1]))
        for i in range(B.shape[1]):
            m = M[:,i] # drop rows where mask is zero
            output[:,i] = np.linalg.lstsq(A[m], B[m,i])[0]
            
    return output


def twophase_mat_sampler(M,r,beta,budget): 
    (n,m) = M.shape
    Omega_1 = round(beta*budget)

    mask = np.zeros(n*m, dtype="bool")
    mask[:Omega_1] = True

    np.random.shuffle(mask)
    mask = mask.reshape((n,m))
    P_Omega = np.zeros_like(M)
    P_Omega[mask] = M[mask]
    U,S,Vt = np.linalg.svd(P_Omega,full_matrices=False)
    mu_A = (n/r)*np.linalg.norm(U[:,:r],axis=1)**2
    nu_A = (m/r)*np.linalg.norm(Vt[:r,:],axis=0)**2


    weights = []
    idx_choice = []
    for i in range (n):
        for j in range(m):
            if mask[i,j] == 1:
                weights.append(0+1/(n*m))
            else:
                weights.append((r/n)*(mu_A[i]+ nu_A[j])*(np.log(n*m)**2)+1/(n*m))

    weights = weights / np.sum(weights)
    #print(weights.size,n*m,round((1-beta)*budget),np.count_nonzero(weights))
    
    draw = np.random.choice(int(n*m), size=round((1-beta)*budget),p=weights, replace=False)
    idxs = []
    for d in draw:
        i = d // m
        j = d % m
        idxs.append((i,j))
        mask[i,j] = True
    
    return mask

#########################
# Monkey Patch of the NucNorm Solver class from fancy impute, the only change is to switch out the solver
#########################
class NuclearNormMinimization(Solver):
    """
    Simple implementation of "Exact Matrix Completion via Convex Optimization"
    by Emmanuel Candes and Benjamin Recht using cvxpy.
    """

    def __init__(
            self,
            require_symmetric_solution=False,
            min_value=None,
            max_value=None,
            error_tolerance=1e-8,
            max_iters=10000,
            verbose=True):
        """
        Parameters
        ----------
        require_symmetric_solution : bool
            Add symmetry constraint to convex problem
        min_value : float
            Smallest possible imputed value
        max_value : float
            Largest possible imputed value
        error_tolerance : bool
            Degree of error allowed on reconstructed values. If omitted then
            defaults to 0.0001
        max_iters : int
            Maximum number of iterations for the convex solver
        verbose : bool
            Print debug info
        """
        Solver.__init__(
            self,
            min_value=min_value,
            max_value=max_value)
        self.require_symmetric_solution = require_symmetric_solution
        self.error_tolerance = error_tolerance
        self.max_iters = max_iters
        self.verbose = verbose

    def _constraints(self, X, missing_mask, S, error_tolerance):
        """
        Parameters
        ----------
        X : np.array
            Data matrix with missing values filled in
        missing_mask : np.array
            Boolean array indicating where missing values were
        S : cvxpy.Variable
            Representation of solution variable
        """
        ok_mask = ~missing_mask
        masked_X = cvxpy.multiply(ok_mask, X)
        masked_S = cvxpy.multiply(ok_mask, S)
        abs_diff = cvxpy.abs(masked_S - masked_X)
        close_to_data = abs_diff <= error_tolerance
        constraints = [close_to_data]
        if self.require_symmetric_solution:
            constraints.append(S == S.T)

        if self.min_value is not None:
            constraints.append(S >= self.min_value)

        if self.max_value is not None:
            constraints.append(S <= self.max_value)

        return constraints

    def _create_objective(self, m, n):
        """
        Parameters
        ----------
        m, n : int
            Dimensions that of solution matrix
        Returns the objective function and a variable representing the
        solution to the convex optimization problem.
        """
        # S is the completed matrix
        shape = (m, n)
        S = cvxpy.Variable(shape, name="S")
        norm = cvxpy.norm(S, "nuc")
        objective = cvxpy.Minimize(norm)
        return S, objective

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        m, n = X.shape
        S, objective = self._create_objective(m, n)
        constraints = self._constraints(
            X=X,
            missing_mask=missing_mask,
            S=S,
            error_tolerance=self.error_tolerance)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(
            verbose=True,
            solver=cvxpy.SCS,
            max_iters=self.max_iters,
            eps = self.error_tolerance,
            # use_indirect, see: https://github.com/cvxgrp/cvxpy/issues/547
            use_indirect=False)
        return S.value
    
def tensor_sandwich(T_true,T_noise,N,r,relnoise,n_slices,alpha,beta,delta,gamma,full_output=False):

    slice_idx = np.random.choice(N,n_slices,replace=False)
    
    budget = min(round(gamma*N*r*np.log(N)**2),round(N**2))


    
    Mask = tl.zeros(T_true.shape,dtype=bool)
    
    completed_slices = tl.zeros(Mask[:,:,:n_slices].shape)
    for i,s in enumerate(slice_idx):
        Mask[:,:,s] = twophase_mat_sampler(T_true[:,:,s],r,beta,budget)
        completed_slices[:,:,i] = NuclearNormMinimization().solve(T_noise[:,:,s],~Mask[:,:,s] )
    T_tilde_3 = tl.unfold(T_noise,2).T
    M_3 = tl.unfold(Mask,2).T    
    
    #random vectors for projecting
    ga = np.random.normal(size=n_slices)
    gb = np.random.normal(size=n_slices)
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
    #contract along mode three using the random vector
    Ta = (completed_slices @ ga).reshape(N,N)
    Tb = (completed_slices @ gb).reshape(N,N) 
    #Now threshold the SVD of Ta and Tb to the given rank
    Ua, Sa, VaT = np.linalg.svd(Ta, full_matrices=0)
    Ta = Ua[:,:r]@ np.diag(Sa[:r]) @ VaT[:r,:]
    Ub, Sb, VbT = np.linalg.svd(Tb, full_matrices=0)
    Tb = Ub[:,:r]@ np.diag(Sb[:r]) @ VbT[:r,:]
    #Using the regularized rank r Ta and Tb, compute X and its eigen decomposition
    X = (np.linalg.pinv(Ta) @ Tb).T
    evalsV, evecsV = np.linalg.eig(X)
    #Find the indices which match the largest eigenvalues by magnitude and sort in the in descending order
    evalsV_idx = np.argsort(-np.abs(evalsV))
    #Pick out the eigenvectors that match
    #LearnedV = evecsV[:,evalsV_idx[:r]]
    LearnedV = evecsV[:,:r]
    #Invert V and use its pseudoinverse to find the matching U
    LearnedU = Ta @ np.linalg.pinv(LearnedV.T)
    #These U may not be normalized (?)
    #LearnedU = LearnedU / np.linalg.norm(LearnedU, axis=0)
    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    Q,R,P = qr(A.T, pivoting=True)
    oversample = round(delta*r)
    M_3[P[:oversample]] = 1
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = censored_lstsq(A, T_tilde_3, M_3).T
    T_recon = tl.cp_to_tensor((np.ones(r), [LearnedU,LearnedV,LearnedW]))
    error = rel_error(T_recon,T_true)
    
    if full_output: return [error,budget,np.sum(M_3),np.sum(M_3)/N**3],(np.ones(r), [LearnedU,LearnedV,LearnedW]), tl.fold(M_3,2,(N,N,N))
    else: return [error,budget,np.sum(M_3),np.sum(M_3)/N**3]

def tensor_sandwich_real(T_true,r,n_slices,beta,delta,gamma,full_output=False):
    N = T_true.shape
    slice_idx = np.random.choice(N[-1],n_slices,replace=False)
    max_N = max(N)
    budget = min(round(gamma*max_N*r*np.log(max_N)**2),round(max_N**2))


    
    Mask = tl.zeros(T_true.shape,dtype=bool)
    
    completed_slices = tl.zeros(Mask[:,:,:n_slices].shape)
    for i,s in enumerate(slice_idx):
        Mask[:,:,s] = twophase_mat_sampler(T_true[:,:,s],r,beta,budget)
        completed_slices[:,:,i] = NuclearNormMinimization().solve(T_true[:,:,s],~Mask[:,:,s] )
    T_tilde_3 = tl.unfold(T_true,2).T
    M_3 = tl.unfold(Mask,2).T    
    
    #random vectors for projecting
    ga = np.random.normal(size=n_slices)
    gb = np.random.normal(size=n_slices)
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
    #contract along mode three using the random vector
    Ta = (completed_slices @ ga).reshape(N[0],N[1])
    Tb = (completed_slices @ gb).reshape(N[0],N[1]) 
    #Now threshold the SVD of Ta and Tb to the given rank
    Ua, Sa, VaT = np.linalg.svd(Ta, full_matrices=0)
    Ta = Ua[:,:r]@ np.diag(Sa[:r]) @ VaT[:r,:]
    Ub, Sb, VbT = np.linalg.svd(Tb, full_matrices=0)
    Tb = Ub[:,:r]@ np.diag(Sb[:r]) @ VbT[:r,:]
    #Using the regularized rank r Ta and Tb, compute X and its eigen decomposition
    X = (np.linalg.pinv(Ta) @ Tb).T
    evalsV, evecsV = np.linalg.eig(X)
    #Find the indices which match the largest eigenvalues by magnitude and sort in the in descending order
    evalsV_idx = np.argsort(-np.abs(evalsV))
    #Pick out the eigenvectors that match
    #LearnedV = evecsV[:,evalsV_idx[:r]]
    LearnedV = evecsV[:,:r]
    #Invert V and use its pseudoinverse to find the matching U
    LearnedU = Ta @ np.linalg.pinv(LearnedV.T)
    #These U may not be normalized (?)
    #LearnedU = LearnedU / np.linalg.norm(LearnedU, axis=0)
    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    Q,R,P = qr(A.T, pivoting=True)
    oversample = round(delta*r)
    M_3[P[:oversample]] = 1
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = censored_lstsq(A, T_tilde_3, M_3).T
    T_recon = tl.cp_to_tensor((np.ones(r), [LearnedU,LearnedV,LearnedW]))
    error = rel_error(T_recon,T_true)
    
    if full_output: return [error,budget,np.sum(M_3),np.sum(M_3)/(N[0]*N[1]*N[2])],(np.ones(r), [LearnedU,LearnedV,LearnedW]), tl.fold(M_3,2,(N,N,N))
    else: return [error,budget,np.sum(M_3),np.sum(M_3)/(N[0]*N[1]*N[2])]
    
def tensor_deli(T_true,
                T_noise,
                r,
                relnoise,
                n_slices,
                alpha,
                beta,
                delta,
                gamma,
                slice_idx=None,
                inner_als=False,
                post_als=False,
                adaptive=True,
                full_output=False,
                ks_complete=False
                ):
    
    d = len(T_noise.shape)
    N = T_true.shape
    total_samples = 0
    #for now, we'll just arbitrarily pick the first index to hold constant for modes after the first three
    sub_tensor3_idx = [list(range(N[0])),list(range(N[1])),list(range(N[2]))] + [[0] for _ in range(d-3)]
    
    T_noise_3 = np.squeeze(T_noise[np.ix_(*sub_tensor3_idx)])

    if slice_idx is None:
        slice_idx = np.random.choice(N[2],n_slices,replace=False)
    else:
        n_slices = len(slice_idx)
        
    print("Selected slices:", slice_idx)
    
    #this is the matrix sampling budget, gamma is just the proportion of the entries to sample, insert whatever theory you want here about what that should be
    budget = round(gamma*N[0]*N[1])
    
    #Mask is going to be updated in phases
    Mask = tl.zeros(T_noise.shape,dtype=bool)
    
    #this is where we'll store the completed slices
    completed_slices = tl.zeros(T_noise_3[:,:,:n_slices].shape)
    
    #loop through the selected slices and complete, updating mask as we go
    if ks_complete == True:
        error_tol = 0.0001
        sample_complexity = [0.8,0.2,0.1]
        U = {}
        sub_tensor3_idx_slice = [list(range(N[0])),list(range(N[1])),slice_idx] + [[0] for _ in range(d-3)]
        for i in range(1,d+1):
            U[i] = []

        completed_slices, U, sample_complexity, total_samples, error_tol = seq_tensor_complete(T_noise[np.ix_(*sub_tensor3_idx_slice)], U, sample_complexity, total_samples, error_tol)
        budget = total_samples
        print("samples after complete KS:", total_samples)
        print("subtensor error:", tl.norm(completed_slices - T_noise[np.ix_(*sub_tensor3_idx_slice)]) / tl.norm(T_noise[np.ix_(*sub_tensor3_idx_slice)]))
        print("shape: ", completed_slices.shape)
    else:
        for i,s in enumerate(slice_idx):
            sub_tensor3_idx_slice = [list(range(N[0])),list(range(N[1])),[s]] + [[0] for _ in range(d-3)]

            if adaptive==True: 
                Mask[np.ix_(*sub_tensor3_idx_slice)] = twophase_mat_sampler(np.squeeze(T_noise[np.ix_(*sub_tensor3_idx_slice)]),r,beta,budget).reshape(Mask[np.ix_(*sub_tensor3_idx_slice)].shape)
            else:
                slice_mask = np.zeros(N[0]*N[1], dtype=bool)
                slice_mask[:budget] = 1
                np.random.shuffle(slice_mask)
                Mask[np.ix_(*sub_tensor3_idx_slice)] = slice_mask.reshape(Mask[np.ix_(*sub_tensor3_idx_slice)].shape)

            print("Completing slice: ", s)
            print("Sample size: ", np.sum( Mask[np.ix_(*sub_tensor3_idx_slice)] ) )

            #call the nuclear norm solver. Insert whatever completion strategy you want here
            completed_slices[:,:,i] = NuclearNormMinimization().solve(np.squeeze(T_noise[np.ix_(*sub_tensor3_idx_slice)]),np.squeeze(~Mask[np.ix_(*sub_tensor3_idx_slice)]))
    
    #Flatten the appropriate three mode tensor
    T_tilde_3 = tl.unfold(T_noise_3,2).T
    
    #This is the method that uses Jennrich to learn the first two factors
    if inner_als==False:
    
        #random vectors for projecting
        ga = np.random.normal(size=n_slices)
        gb = np.random.normal(size=n_slices)
        ga = ga / np.linalg.norm(ga)
        gb = gb / np.linalg.norm(ga)
        
        #contract along mode three using the random vector
        Ta = (completed_slices @ ga).reshape(N[0],N[1])
        Tb = (completed_slices @ gb).reshape(N[0],N[1]) 
        
        #Now threshold the SVD of Ta and Tb to the given rank
        Ua, Sa, VaT = np.linalg.svd(Ta, full_matrices=0)
        Ta = Ua[:,:r]@ np.diag(Sa[:r]) @ VaT[:r,:]
        Ub, Sb, VbT = np.linalg.svd(Tb, full_matrices=0)
        Tb = Ub[:,:r]@ np.diag(Sb[:r]) @ VbT[:r,:]
        
        #Using the regularized rank r Ta and Tb, compute X and its eigen decomposition
        X = (np.linalg.pinv(Ta) @ Tb).T
        evalsV, evecsV = np.linalg.eig(X)
        LearnedV = evecsV[:,:r]
        
        #Invert V and use its pseudoinverse to find the matching U
        LearnedU = Ta @ np.linalg.pinv(LearnedV.T)
    
    #If you don't want to use Jennrich, you could just use ALS on the subproblem
    else:
        inner_cp = parafac(completed_slices, r, verbose=True)
        LearnedU = inner_cp[1][0]
        LearnedV = inner_cp[1][1]
    
    #Now set up censored least square to learn the third factor matrix
    A = khatri_rao(LearnedU, LearnedV)
    
    #setup of the mask by filling in ones for the fibers

    
    oversample = round(delta*r)
    M_3 = np.squeeze(Mask[np.ix_(*sub_tensor3_idx)])
    M_3 = tl.unfold(M_3,2).T
    
    if adaptive==True:
        #Do QR to select the rows that ensure the system is consistent. Oversample using delta*rank
        Q,R,P = qr(A.T, pivoting=True)
        #choosing randomly after the rank is a way to avoid grabbing long consecutive clumps of indices, which is probably not desirable
        rows_to_sample = np.concatenate( (P[:r], np.random.choice(P[r:],oversample-r, replace=False)))

        M_3[rows_to_sample] = 1
    else:
        inter_slice_mask = np.zeros(M_3.size, dtype=bool)
        inter_slice_mask[:oversample*N[2]] = 1
        np.random.shuffle(inter_slice_mask)
        M_3 = np.bitwise_or(inter_slice_mask.reshape(M_3.shape), M_3)
            
        
    #solve for the third factor matrix
    LearnedW = censored_lstsq(A, T_tilde_3, M_3).T
    
    #Update the global mask
    Mask[np.ix_(*sub_tensor3_idx)] =  tl.fold(M_3.T,2,(N[0],N[1],N[2])).reshape(Mask[np.ix_(*sub_tensor3_idx)].shape)
    
    #now loop through and solve the remaining factor matrices beyond the first 3
    learned_factors = [LearnedU,LearnedV,LearnedW]
    if d > 3:
        for k in range(3,d):
            print("Solving Mode: ", k)
            
            #setup the indices for the next 3 mode subtensor [N,N,N,0] -> [N,N,0,N]
            sub_tensor3_idx[k] = list(range(N[k]))
            sub_tensor3_idx[k-1] = [0]        
            
            #setup up the approprioate flattened version of the subtensor. The missing mode should always be mode 2 and modes 0 and 1 are fixed
            T_noise_k = np.squeeze(T_noise[np.ix_(*sub_tensor3_idx)])
            T_tilde_k = tl.unfold(T_noise_k,2).T
            
            Mask_k = np.squeeze(Mask[np.ix_(*sub_tensor3_idx)])
            Mask_k = tl.unfold(Mask_k,2).T
            if adaptive==True:
                #Using the same pivots as before
                Mask_k[rows_to_sample] = 1
            else:
                inter_slice_mask = np.zeros(Mask_k.size, dtype=bool)
                inter_slice_mask[:(oversample*N[k])] = 1
                np.random.shuffle(inter_slice_mask)
                Mask_k = np.bitwise_or(inter_slice_mask.reshape(Mask_k.shape), Mask_k)

            #Solve it
            LearnedX = censored_lstsq(A, T_tilde_k, Mask_k).T
            
            #update global mask, tack on latest factor
            Mask[np.ix_(*sub_tensor3_idx)] = tl.fold(Mask_k.T,2,(N[0],N[1],N[k])).reshape(Mask[np.ix_(*sub_tensor3_idx)].shape)
            learned_factors.append(LearnedX)
            
            #this is probably not necessary, added because of some OOM problems I had
            del T_tilde_k, Mask_k

    #final solve to fix the weights
    best_weights =  censored_lstsq(tl.tenalg.khatri_rao(learned_factors), T_noise.reshape(-1), Mask.reshape(-1))
    T_recon = tl.cp_to_tensor((best_weights,learned_factors))
    
    #gather results and return them
    #error = rel_error(T_recon,T_true)
    error = tl.norm(T_recon - T_true) / tl.norm(T_true)
    total_samples += np.sum(Mask)   
    
        
    if full_output: return [error,budget,total_samples,total_samples/T_true.size,(best_weights, learned_factors),Mask]
    else: return [error,budget,total_samples,total_samples/T_true.size]
    
def tensor_als(T_true,T_noise,N,r,relnoise,budget,alpha,max_iter,init='svd',mask=None):

    if mask is None:
        sample_mask = np.zeros(T_true.size, dtype="bool")
        sample_mask[:int(budget)] = True
        np.random.shuffle(sample_mask)

        sample_mask = sample_mask.reshape(T_true.shape)
    else: 
        sample_mask = mask
        
    res = parafac(T_noise,rank=r, n_iter_max=max_iter,init=init,mask=sample_mask)
    T_recon = tl.cp_to_tensor(res)
    error = rel_error(T_recon,T_true)
    
    return [error]

def construct_proj(U):
    if len(U.shape) == 1:
        U = U.reshape(-1,1)
    return U @ np.linalg.pinv(U.T @ U) @ U.T

def seq_tensor_complete(T, U, sample_complexity, total_samples, error_tol):

    n_modes = len(T.shape)
    dims = T.shape

    if len(U[n_modes]) == 0:
        if n_modes == 1:
            U[n_modes] = (T / tl.norm(T)).reshape(-1,1)
            
            total_samples += T.size
        else:
            T_est = []
            for j in range(dims[-1]):
                sub_idx = [list(range(dims[i])) for i in range(n_modes-1)] + [[j]]
                #print(sub_idx, n_modes, T.shape)
                T_j = np.squeeze(T[np.ix_(*sub_idx)])
                T_j_est,U, sample_complexity, total_samples, error_tol = seq_tensor_complete(T_j, U, sample_complexity, total_samples, error_tol)
                T_est.append(T_j_est)
                
            T = np.stack(T_est,axis=n_modes - 1)
            U[n_modes] = (T / np.linalg.norm(T.reshape(-1))).reshape(-1,1)
        return T, U, sample_complexity, total_samples, error_tol
    else:
        sub_size = T.size
        Mask = np.zeros(sub_size, dtype=bool)
        number_samples = round(sub_size*sample_complexity[n_modes-1])
        Mask[:number_samples] = 1
        np.random.shuffle(Mask)
        U_Omega = (U[n_modes] * Mask[:, None])
        
        
        #projector matrix for restricted basis 
        P_U_Omega = construct_proj(U_Omega)

        #data is also masked
        T_Omega = np.multiply(Mask, T.reshape(-1))
        
        total_samples +=number_samples
        
        #What's the distance from the data projected onto our basis vs the data as we see it
        Residual = T_Omega - P_U_Omega @ T_Omega

        #print("residual: ", np.linalg.norm(Residual))

        #if its big, we need to add the normalized fiber to the basis, this means sampling the whole thing
        if np.linalg.norm(Residual) > error_tol:           
            if n_modes > 1:
                T_est = []
                for j in range(dims[-1]):
                    sub_idx = [list(range(dims[i])) for i in range(n_modes-1)] + [[j]]
                    #print(sub_idx, n_modes, T.shape)
                    T_j = np.squeeze(T[np.ix_(*sub_idx)])
                    T_j_est, U, sample_complexity, total_samples, error_tol = seq_tensor_complete(T_j, U, sample_complexity, total_samples, error_tol)
                    T_est.append(T_j_est)
                T = np.stack(T_est,axis=n_modes - 1)
            else:
                total_samples += (T.size - number_samples)
                          
            U_perp = np.eye(U[n_modes].shape[0]) - construct_proj(U[n_modes])
            new_col = U_perp @ T.reshape(-1)
            new_col = new_col / np.linalg.norm(new_col)
            
          
            U[n_modes] = np.hstack([U[n_modes], new_col.reshape(-1,1)])

            U_Omega = U[n_modes] * Mask[:, None]
            
        T = U[n_modes] @ np.linalg.pinv(U_Omega.T @ U_Omega)@ U_Omega.T @ np.multiply(Mask, T.reshape(-1))
        if n_modes > 1:
            T = T.reshape(dims)
        return T, U, sample_complexity, total_samples, error_tol

def tucker_core_to_cp(TuckerSpaces, CoreCPSpaces):
    """Convenience function which takes the rxr CPD factors obtained from a (rxrxr) core tensor from a Tucker decomp, along with the Nxr Tucker factor matrices and
    're-inflates' these to the full NxNxN tensor with a rank r CPD. This is a trick Haselby observed for speeding up the Tucker to CPD step of the overall algorithm
    
    ----
    TuckerSpaces (list) : List of Nxr matrices that are the factor matrices in a Tucker decomp
    CoreCPSpaces (list) : list of rxr matrices that are the factor matrices of a CPD of the core tensor of a Tucker decomp
    -------
    K (array) : NxNxN tensor with CPD of rank r
    """
    N = TuckerSpaces[0].shape[0]
    r = TuckerSpaces[0].shape[1]

    K = khatri_rao(TuckerSpaces[0] @ CoreCPSpaces[0], TuckerSpaces[1] @ CoreCPSpaces[1])
    K = khatri_rao(K, TuckerSpaces[2] @ CoreCPSpaces[2])
    K = K@np.ones(r)
    return K.reshape(N,N,N)
  
def core_censored_solve(T,F,sample_mask,N,r):
    """Solves censored least squares problem to find the core of a Tucker, given factors and a tensor with missing values according to sample mask.
    ----
    T (array) : 3-way tensor assumed to be cube of size NxNxN that has missing data as described by the given mask
    F (list) : List of Nxr arrays that are estimates for the subspaces spanned by the true factor matrices of the tensor
    sample_mask (array) : 1D boolean array with N^3 entries. when reshaped, should be the sample pattern for the missing values
    N (int) : dimension of any one of the modes of the tensor
    r (int) : rank
    -------
    Tucker_T (array) : Tensor NxNxN with tucker decomposition with factors F and best fit core according to censored least squares

    """
    Projector = tl.tenalg.kronecker(F)
    Core = censored_lstsq(Projector, T.reshape(-1), sample_mask)
    return (Projector@Core).reshape(N,N,N), Core.reshape(r,r,r)

def initilization(T,sample_mask,N,p,r,c,mu,sigma_1,sigma_r):
    """ Given a tensor where values are missing per the sample mask find an estimate for the subspaces spanned by its factor matrices
    ----
    T (array) : 3-way tensor assumed to be cube of size NxNxN
    sample_mask : 1D boolean array with N^3 entries. when reshaped, should be the sample pattern for the missing values
    N (int) : dimension of any one of the modes of the tensor
    p (float) :  sample rate of the tensor
    r (int) : assumed rank of the tensor, user must decide
    c (float) : c*sqrt(r/N) bounds the row norm of all factor matrices
    mu (float) :  coherence bound for all the factor matrices
    sigma_1 (float) : largest singular value of any factor matrix
    sigma_r (float) : rank-th smallest singular value. In exact setting would be the smallest nonzero singular value if the tensor is truly rank r.
    -------
    F (list) : List of Nxr arrays that are estimates for the subspaces spanned by the true factor matrices of the tensor

    This algorithm is based on Montarri and Sun, reused by Moitra
    """
    #In this intialization, missing values are zeroed 
    T0 = T.reshape(-1).copy()
    T0[~sample_mask] = 0
    T0 = T0.reshape(N,N,N)

    #number of modes
    M = len(T0.shape)
    F = []
    #tau parameter is used to control the coherence of the estimated factor matrices by zeroing out bad rows
    tau = np.sqrt(r/N) * ((2*mu*r*sigma_1**2) / ((c**2)*(sigma_r**2)))**5
    for m in range(M):
        U = tl.unfold(T0,m)
        D = np.diag(np.diag(U @ U.T))
        B = (1/p)*D + (1/p**2)*(U @ U.T - D)

        #Zero out rows in the factor matrix which have a norm that is larger than the tau number calculated
        Ur, _, _ = np.linalg.svd(B)
        X = Ur[:,:r]
        r_norms = np.linalg.norm(X, axis=1)
        X[r_norms > tau] = 0
        
        #Just need a set of orthonormal basis elements for the space spanned bu X, Q from QR will do the trick
        Qr, _= np.linalg.qr(X)

        F.append(Qr)

    return F

def kron_alt_min(T,F,N,r,sample_mask,p,sigma_1,sigma_r,k=10):
    """ Given a tensor where values are missing per the sample mask and an estimate of the factor matrices find a (better) estimate of the subspaces spanned by its factor matrices
    ----
    T (array) : 3-way tensor assumed to be cube of size NxNxN
    F (list) : List of three factor matrices for each of the modes of size Nxr
    sample_mask (array) : 1D boolean array with N^3 entries. when reshaped, should be the sample pattern for the missing values
    N (int) : dimension of any one of the modes of the tensor
    r (int) : assumed rank of the tensor, user must decide
    p (float) :  sample rate of the tensor
    sigma_1 (float) : largest singular value of any factor matrix
    sigma_r (float) : rank-th smallest singular value. In exact setting would be the smallest nonzero singular value if the tensor is truly rank r.
    sub_sample (float) : default is 0.5. Subsample rate is used to vary which samples are considered in each run of outer loop. Helps performance, and may help avoid getting stuck in local mins
    k (int) : number of iterations to run
    -------
    F (list) : List of Nxr arrays that are estimates for the subspaces spanned by the true factor matrices of the tensor

    This algorithm is described in Moitra, Liu. It is essentially a Tucker decomposition that uses censored least square in the inner loop
    """

    #Here is the theoritcal number of iterations you'd need to run
    #to get the exact results stated in paper. It is way too large to use in practice I have discovered
    #k = int( 100*np.log((N*sigma_1) / (c*sigma_r) ) )
    
    #subsample rate for their proofs, also found this to be bad in practice
    #p_prime = p / k

    for t in range(k):
        V = F
        M = sample_mask.reshape((N,N,N))
        for m in range(3):
            modes = list(range(3))
            modes.remove(m)
            L=modes[0]
            R=modes[1]
            Mm = tl.unfold(M,m)
            B = np.linalg.qr(np.kron(V[L],V[R]))[0].T

            H = censored_lstsq(B.T,tl.unfold(T,m).T,Mm.T)
            Ur, _, _ = np.linalg.svd(H.T)
            F[m] = Ur[:,:r]

    return F

def iwen_jennrich(T_tilde,r):
    N = T_tilde.shape
    T_tilde_3 = tl.unfold(T_tilde,2).T
    
    #random vectors for projecting
    ga = np.random.normal(size=N[2])
    gb = np.random.normal(size=N[2])
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
       
    #contract along mode three using the random vector
    Ta = (T_tilde_3 @ ga).reshape(N[0],N[1])
    Tb = (T_tilde_3 @ gb).reshape(N[0],N[1]) 
    
    #Now threshold the SVD of Ta and Tb to the given rank
    Ua, Sa, VaT = np.linalg.svd(Ta, full_matrices=0)
    Ta = Ua[:,:r]@ np.diag(Sa[:r]) @ VaT[:r,:]

    Ub, Sb, VbT = np.linalg.svd(Tb, full_matrices=0)
    Tb = Ub[:,:r]@ np.diag(Sb[:r]) @ VbT[:r,:]

    #Using the regularized rank r Ta and Tb, compute X and its eigen decomposition
    X = (np.linalg.pinv(Ta) @ Tb).T
    evalsV, evecsV = np.linalg.eig(X)
    
    #Find the indices which match the largest eigenvalues by magnitude and sort in the in descending order
    evalsV_idx = np.argsort(-np.abs(evalsV))
    #Pick out the eigenvectors that match
    LearnedV = evecsV[:,evalsV_idx[:r]]

    #Invert V and use its pseudoinverse to find the matching U
    LearnedU = Ta @ np.linalg.pinv(LearnedV.T)

    #These U may not be normalized (?)
    #LearnedU = LearnedU / np.linalg.norm(LearnedU, axis=0)
    
    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = res[0].T

    IwenFactors = [LearnedU, LearnedV, LearnedW]
    
    return IwenFactors

def org_jennrich(T_tilde,N,r):
    """ Given a tensor finds the CPD of a given rank based on Jennrich's algorithm
    ----
    T_tilde (array) : 3-way tensor assumed to be cube of size NxNxN. 

    N (int) : dimension of any one of the modes of the tensor
    r (int) : assumed rank of the tensor, user must decide
    -------
    JenFactors (list) : List of Nxr arrays that are matrices of the tensor for a CPD of rank r. In the exact setting and with assumptions about the linear independence of components, would be exact

    This algorithm is described orginally by Harshman, and appeared many times in the literature since
    """

    T_tilde_3 = tl.unfold(T_tilde,2).T
 
    #random vectors for projecting
    ga = np.random.normal(size=N)
    gb = np.random.normal(size=N)
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
       
    #contract along mode three using the random vector
    Ta = (T_tilde_3 @ ga).reshape(N,N)
    Tb = (T_tilde_3 @ gb).reshape(N,N) 
    

    X = Ta @ np.linalg.pinv(Tb)
    Y = (np.linalg.pinv(Ta) @ Tb).T

    #compute the eigen decompositions of X any Y    
    evalsU, evecsU = np.linalg.eig(X)
    evalsV, evecsV = np.linalg.eig(Y)
    
    #Find the sorting of the eigenvalues by magnitude
    evalsU_idx = np.argsort(np.abs(evalsU))
    evalsV_idx = np.argsort(np.abs(evalsV))
    
    #Truncate to the correct rank, and flip one index so reciprocals will match
    matchedV_idx = evalsV_idx[N-r:]
    if (N-r)==0:
        matchedU_idx = np.flip(evalsU_idx)
    else:
         matchedU_idx = evalsU_idx[-1:(N-r-1):-1]
            
    #Pick out the corresponding eigenvectors
    LearnedU = evecsU[:,matchedU_idx]
    LearnedV = evecsV[:,matchedV_idx]

    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = res[0].T

    JenFactors = [LearnedU, LearnedV, LearnedW]
    return JenFactors