
"""
25 MAY 2023

This is a script is for running a job and gathering and saving results into a dataframe and csv 

author: Cullen Haselby 
"""
#########################
# IMPORTS
#########################

import csv
from datetime import datetime
import tensor_sandwich
import sys
import timeit
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from random import randint, randrange
import scipy.io
import pickle
###########################################################
# PARSE INPUTS, RUN TRIALS, STORE RESULTS IN CSV
##########################################################


if __name__ == "__main__":
    
    #Take the command line arguments, get rid of the script name
    args = sys.argv[1:]

    #store the settings into a list, splitting and casting as ints

    ts=[int(t) for t in args[0].split(",")]
    ns=[int(n) for n in args[1].split(",")]
    rs=[int(r) for r in args[2].split(",")]
    noises=[float(eps) for eps in args[3].split(",")]
    mode = [m for m in args[4].split(",")]
    
    now = datetime.now()
    dt_string = now.strftime("%m%d%H%M%S")
    name="results/"+mode[0]+"/results_"+dt_string+"_"+str(randint(100, 999))+".csv"
    
    #organize the parameters into a list of tuples
    if mode[0] == 'sandwich':
        n_slices = [int(s) for s in args[5].split(",")]
        alphas=[int(a) for a in args[6].split(",")]
        betas=[float(b) for b in args[7].split(",")]
        deltas=[float(d) for d in args[8].split(",")]
        gammas=[float(g) for g in args[9].split(",")]

        params=[(t,n,r,eps,s,a,b,d,g) for t in ts for n in ns for r in rs for eps in noises for s in n_slices for a in alphas for b in betas for d in deltas for g in gammas]
        cols=["trial","n","r","SNR","mode","slices","alpha","beta","delta","gamma","time","rel_error","matrix budget","total budget", "revealed"]
        

        print("Writing results to ", name)
        with open(name, 'a', newline='') as f_object:  
            # Pass the CSV  file and write in the column headers
            write = csv.writer(f_object)
            write.writerow(cols)  
            f_object.close()
        
        for p in params:
            results = []
            trials,n,r,eps,s,a,b,d,g = p
            print("Running trial(s) for parameters ", p)
            
            for t in range(trials):
                if eps == 0: SNR = np.inf
                else: 
                    SNR = round(10*np.log(1/eps)/np.log(10))
                    
                weights = [1/p**a for p in range(1,r+1)]
                T_true = tl.random.random_cp((n,n,n),r)
                T_true[0] = weights
                T_true = tl.cp_to_tensor(T_true)

                T_noise = tensor_sandwich.add_noise(T_true,eps)
                
                starttime = timeit.default_timer()
                res = tensor_sandwich.tensor_sandwich(T_noise,T_true,n,r,eps,s,a,b,d,g)
                recover_time = timeit.default_timer() - starttime
                results.append([t,n,r,SNR,mode[0],s,a,b,d,g,recover_time] + res)
                
                print(results)
    elif mode[0] == 'sand_als':
        n_slices = [int(s) for s in args[5].split(",")]
        alphas=[int(a) for a in args[6].split(",")]
        betas=[float(b) for b in args[7].split(",")]
        deltas=[float(d) for d in args[8].split(",")]
        gammas=[float(g) for g in args[9].split(",")]
        #budget = [int(b) for b in args[6].split(",")]
        max_iter=[int(i) for i in args[10].split(",")]
        
        params=[(t,n,r,eps,s,a,b,d,g,mi) for t in ts for n in ns for r in rs for eps in noises for s in n_slices for a in alphas for b in betas for d in deltas for g in gammas for mi in max_iter]
        cols=["trial","n","r","SNR","mode","slices","alpha","beta","delta","gamma","max_iter","time","rel_error_sandwich","matrix budget","total budget", "revealed", "rel_error_post_als","rel_error_just_als"]
        

        print("Writing results to ", name)
        with open(name, 'a', newline='') as f_object:  
            # Pass the CSV  file and write in the column headers
            write = csv.writer(f_object)
            write.writerow(cols)  
            f_object.close()
        
        for p in params:
            results = []
            trials,n,r,eps,s,a,b,d,g,mi = p
            print("Running trial(s) for parameters ", p)
            
            for t in range(trials):
                if eps == 0: SNR = np.inf
                else: 
                    SNR = round(10*np.log(1/eps)/np.log(10))
                    
                weights = [1/p**a for p in range(1,r+1)]
                T_true = tl.random.random_cp((n,n,n),r)
                T_true[0] = weights
                T_true = tl.cp_to_tensor(T_true)

                T_noise = tensor_sandwich.add_noise(T_true,eps)

                starttime = timeit.default_timer()
                res_sand, T_sand, mask = tensor_sandwich.tensor_sandwich(T_noise,T_true,n,r,eps,s,a,b,d,g,full_output = True)
                #T_sand = res_sand[1]
                res_als = tensor_sandwich.tensor_als(T_true,T_noise,n,r,eps,0,a,mi,init=T_sand,mask=mask)
                res_als_alone = tensor_sandwich.tensor_als(T_true,T_noise,n,r,eps,res_sand[-2],a,mi,init='svd',mask=None)
                recover_time = timeit.default_timer() - starttime
                results.append([t,n,r,SNR,mode[0],s,a,b,d,g,mi,recover_time] + res_sand+res_als+res_als_alone)
                
                print(results)                
    elif mode[0] == 'kron_alt':
        alphas=[int(a) for a in args[5].split(",")]
        n_slices = int(args[6])
        beta= float(args[7])
        delta=float(args[8])
        gamma=float(args[9])
        
        max_iters = [int(mi) for mi in args[10].split(",")]

        params=[(t,n,r,eps,a,mi,n_slices,beta,delta,gamma) for t in ts for n in ns for r in rs for eps in noises for a in alphas for mi in max_iters]
        
        cols=["trial","n","r","SNR","mode","alpha","slices","beta","delta","gamma","max_iter","total budget","revealed","time","rel_error_sand","rel_error_tc_final"]
        
        print("Writing results to ", name)
        with open(name, 'a', newline='') as f_object:  
        # Pass the CSV  file and write in the column headers
            write = csv.writer(f_object)
            write.writerow(cols)  
            f_object.close()
                
        for p in params:
            results = []
            trials,n,r,eps,a,mi,n_slices,beta,delta,gamma = p
            print("Running trial(s) for parameters ", p)
            
            for t in range(trials):
                if eps == 0: SNR = np.inf
                else: 
                    SNR = round(10*np.log(1/eps)/np.log(10))
                    
                weights = [1/p**a for p in range(1,r+1)]
                
                sigma_1 = weights[0]
                sigma_r = weights[-1]
    
                T_true = tl.random.random_cp((n,n,n),r)
                T_true[0] = weights
        
                true_factors = T_true[1]
                T_true = tl.cp_to_tensor(T_true)
                T_noise = tensor_sandwich.add_noise(T_true,eps)
                                
                min_sigma = np.inf
                max_mu = 0
                for factor in true_factors:
                    U,S,Vt = np.linalg.svd(factor)
                    r_norms = np.linalg.norm(factor,axis=1,ord=2)
                    c = np.max(r_norms)*np.sqrt(n/r)
                    mu = (c**2)*r / (S[-1]**2)
                    if mu > max_mu: max_mu = mu
                    if S[-1] < min_sigma: min_sigma = S[-1]                   
                    
               ##################### INITIALIZE ####################
                starttime = timeit.default_timer()

                res_sand, T_sand, sand_mask = tensor_sandwich.tensor_sandwich(T_noise,T_true,n,r,eps,n_slices,a,beta,delta,gamma,full_output = True)
                
                budget = res_sand[2]
                sample_mask = np.zeros(n**3, dtype="bool")
                sample_mask[:budget] = True
                np.random.shuffle(sample_mask)
                prop_revealed = budget / n**3
                    
                SunFactors = tensor_sandwich.initilization(T_noise.copy(),sample_mask,n,prop_revealed,r,c,mu,sigma_1,sigma_r)
                SunT, SunCore = tensor_sandwich.core_censored_solve(T_noise,SunFactors,sample_mask,n,r)

                ##################### KRON ALT MIN #####################
                KronFactors = tensor_sandwich.kron_alt_min(T_noise,SunFactors,n,r,sample_mask,prop_revealed,sigma_1,sigma_r,k=mi)
                KronT, KronCore = tensor_sandwich.core_censored_solve(T_noise,KronFactors,sample_mask,n,r)


                ##################### CP STEP #####################    
                JenFactors = tensor_sandwich.org_jennrich(KronCore.reshape((r,r,r)),r,r)
                #JenFactors = tensor_sandwich.iwen_jennrich(KronCore.reshape((r,r,r)),r,r)
                JenCP = tensor_sandwich.tucker_core_to_cp(KronFactors,JenFactors)
                recover_time = timeit.default_timer() - starttime
                 
                #res_sun = tensor_sandwich.rel_error(SunT,T_true)
                #res_kron = tensor_sandwich.rel_error(KronT,T_true)
                res_final_tc = tensor_sandwich.rel_error(JenCP,T_true)
                
                results.append([t,n,r,SNR,mode[0],a,n_slices,beta,delta,gamma,mi,budget,prop_revealed,recover_time] + [res_sand[0],res_final_tc])
                   
        
                print(results)
    elif mode[0] == 'deli':

        order = [int(d) for d in args[5].split(",")]
        n_slices = [int(s) for s in args[6].split(",")]
        alphas=[int(a) for a in args[7].split(",")]
        betas=[float(b) for b in args[8].split(",")]
        deltas=[float(delt) for delt in args[9].split(",")]
        gammas=[float(g) for g in args[10].split(",")]
        adapt =[bool(int(a)) for a in args[11].split(",")]
        in_als =[bool(int(als)) for als in args[12].split(",")]
        post_als =[bool(int(als)) for als in args[13].split(",")]
        max_iters = [int(a) for a in args[14].split(",")]
        ks_complete = [bool(int(a)) for a in args[15].split(",")]
        #budget = [int(b) for b in args[6].split(",")]

        
        params=[(t,n,r,d,eps,s,a,b,delt,g,ad,als,post,max_iter,ks) for t in ts for n in ns for r in rs for d in order for eps in noises for s in n_slices for a in alphas for b in betas for delt in deltas for g in gammas for ad in adapt for als in in_als for post in post_als for max_iter in max_iters for ks in ks_complete]
        
        cols=["trial","n","r","order","SNR","mode","slices","alpha","beta","delta","gamma","time","adaptive","inner_als","rel_error","matrix budget","total budget","revealed"]
        if post_als[0]==True:
            cols=cols + ["post_als_rel_error", "just_als_rel_error"]

        print("Writing results to ", name)
        with open(name, 'a', newline='') as f_object:  
            # Pass the CSV  file and write in the column headers
            write = csv.writer(f_object)
            write.writerow(cols)  
            f_object.close()
        
        for p in params:
            results = []
            trials,n,r,d,eps,s,a,b,delt,g,ad,als,post_als,max_iter,ks = p
            print("Running trial(s) for parameters ", p)

            for t in range(trials):
                if eps == 0: SNR = np.inf
                else: 
                    SNR = round(10*np.log(1/eps)/np.log(10))

                weights = [1/p**a for p in range(1,r+1)]
                T_true = tl.random.random_cp((n,)*d,r)
                T_true[0] = weights
                T_true = tl.cp_to_tensor(T_true)

                T_noise = tensor_sandwich.add_noise(T_true,eps)

                starttime = timeit.default_timer()
                res = tensor_sandwich.tensor_deli(T_true,T_noise,r,eps,s,a,b,delt,g, adaptive=ad, inner_als=als , full_output = True, post_als=post_als, ks_complete=ks)
                if post_als==True:
                    
                    
                    
                    init_recovered_factors = [np.copy(res[4][1][i]) for i in range(len(T_true.shape))]
                    init_recovered_weights = np.copy(res[4][0])
                    
                    post_als_result = parafac(T_noise,r,init=(init_recovered_weights, init_recovered_factors),n_iter_max = max_iter,mask=res[5],verbose=1)
                    T_est = tl.cp_to_tensor(post_als_result)
                    rel_error = tl.norm(T_est - T_true) / tl.norm(T_true)
                    
                    recover_time = timeit.default_timer() - starttime
                
                    just_als_result = parafac(T_noise,r,n_iter_max = max_iter,mask=res[5],verbose=1)
                    T_est = tl.cp_to_tensor(just_als_result)
                    rel_error_just_als = tl.norm(T_est - T_true) / tl.norm(T_true)
                    res =  res[:4] + [rel_error,rel_error_just_als]

                results.append([t,n,r,d,SNR,mode[0],s,a,b,delt,g,recover_time,ad,als] + res)
                
                print(results)
                print("error:", res[0])
    elif mode[0] == 'seq':

        order = [int(d) for d in args[5].split(",")]
        alphas=[int(a) for a in args[6].split(",")]
        sample_complexity=[float(a) for a in args[7].split(",")]
        error_tols=[float(a) for a in args[8].split(",")] 
        post_als =[bool(int(als)) for als in args[9].split(",")]
        max_iters = [int(a) for a in args[10].split(",")]
        
        params=[(t,n,r,d,eps,a,e,post,max_iter) for t in ts for n in ns for r in rs for d in order for eps in noises for a in alphas for e in error_tols for post in post_als for max_iter in max_iters]
        cols=["trial","n","r","order","SNR","mode","alpha","sampling","tolerance","time","rel_error","revealed"]
        if post_als[0]==True:
            cols= cols + ["post_als_rel_error"]
            
        print("Writing results to ", name)
        with open(name, 'a', newline='') as f_object:  
            # Pass the CSV  file and write in the column headers
            write = csv.writer(f_object)
            write.writerow(cols)  
            f_object.close()
        
        for p in params:
            results = []
            trials,n,r,d,eps,a,error_tol,post_als,max_iter = p
   
            print("Running trial(s) for parameters ", p)
            
            for t in range(trials):
                total_samples = 0
            
                if eps == 0: SNR = np.inf
                else: 
                    SNR = round(10*np.log(1/eps)/np.log(10))

                weights = [1/p**a for p in range(1,r+1)]
                T_true = tl.random.random_cp((n,)*d,r)
                T_true[0] = weights
                T_true = tl.cp_to_tensor(T_true)

                T_noise = tensor_sandwich.add_noise(T_true,eps)

                starttime = timeit.default_timer()
                
                U = {}
                for i in range(1,d+1):
                    U[i] = []
                
                T_est,U, sample_complexity, total_samples, error_tol = tensor_sandwich.seq_tensor_complete(T_noise, U, sample_complexity, total_samples, error_tol)
              

                recover_time = timeit.default_timer() - starttime
                
                rel_error = tl.norm(T_est - T_true) / tl.norm(T_true)
                prop_revealed = total_samples /T_true.size
                
                if post_als==True:
                    post_als_result = parafac(T_est,r,n_iter_max = max_iter,verbose=1)
                    T_est = tl.cp_to_tensor(post_als_result)
                    rel_error_als = tl.norm(T_est - T_true) / tl.norm(T_true)
              
                    
                    results.append([t,n,r,d,SNR,mode[0],a,sample_complexity,error_tol,recover_time,rel_error,prop_revealed,rel_error_als])
                else:
                    results.append([t,n,r,d,SNR,mode[0],a,sample_complexity,error_tol,recover_time,rel_error,prop_revealed]) 
                
                print(results)   
    elif mode[0] == 'big_fluor':

    
        slice_idx = [int(s) for s in args[5].split(",")]
        betas=[float(b) for b in args[6].split(",")]
        deltas=[float(delt) for delt in args[7].split(",")]
        gammas=[float(g) for g in args[8].split(",")]
        adapt =[bool(int(a)) for a in args[9].split(",")]
        in_als =[bool(int(als)) for als in args[10].split(",")]
        #budget = [int(b) for b in args[6].split(",")]
        
        
        params=[(t,r,b,delt,g,ad,als) for t in ts for r in rs for b in betas for delt in deltas for g in gammas for ad in adapt for als in in_als]

        
        for p in params:
            results = []
            trials,r,b,delt,g,ad,als = p
            print("Running trial(s) for parameters ", p)
                
            mat = scipy.io.loadmat('data/fluordata.mat',simplify_cells=True)
            fluor = mat['fluor']['data']

            emission_ticks = mat['fluor']['axisscale'][1][0]
            excite_ticks =  mat['fluor']['axisscale'][2][0]
            class_set = mat['fluor']['class'][0][0][1]
            type_set = mat['fluor']['class'][0][0][0]


            data = fluor
            data = data[:,:,:,0]
            print("Shape of the fluor data: ", data.shape)


            results.append(tensor_sandwich.tensor_deli(data,
                                       data,
                                       r,
                                       0,
                                       len(slice_idx),
                                       1,
                                       b,
                                       delt,
                                       g,
                                       adaptive=ad,inner_als=als,slice_idx=slice_idx,full_output=True))
                           
        name = "results/"+mode[0]+"/results_bigflour"+dt_string+"_"+str(r)+'.pickle'
        print("Writing results to ", name)
        with open(name, 'wb') as f:
            pickle.dump(results, f)
            f.close()               

    else:
        alphas=[int(a) for a in args[5].split(",")]
        budget = [int(b) for b in args[6].split(",")]
        max_iter=[int(i) for i in args[7].split(",")]
        cols=["","r","SNR","mode","total budget","alpha","max_iter","time","rel_error"]

        params=[(t,n,r,eps,b,a,mi) for t in ts for n in ns for r in rs for eps in noises for b in budget for a in alphas for mi in max_iter]
        print("Writing results to ", name)
        with open(name, 'a', newline='') as f_object:  
            # Pass the CSV  file and write in the column headers
            write = csv.writer(f_object)
            write.writerow(cols)  
            f_object.close()

        for p in params:
            results = []
            trials,n,r,eps,budget,a,mi = p
            print("Running trial(s) for parameters ", p)


            for t in range(trials):
                if eps == 0: SNR = np.inf
                else: 
                    SNR = round(10*np.log(1/eps)/np.log(10))

                weights = [1/p**a for p in range(1,r+1)]
                T_true = tl.random.random_cp((n,n,n),r)
                T_true[0] = weights
                T_true = tl.cp_to_tensor(T_true)

                T_noise = tensor_sandwich.add_noise(T_true,eps)

                starttime = timeit.default_timer()
                res = tensor_sandwich.tensor_als(T_true,T_noise,n,r,eps,budget,a,mi)

                recover_time = timeit.default_timer() - starttime

                results.append([t,n,r,SNR,mode[0],budget,a,mi,recover_time] + res)

                print(results)
                        
    with open(name, 'a+', newline='') as f_object:
        write = csv.writer(f_object)
        write.writerows(results)
        f_object.close()
