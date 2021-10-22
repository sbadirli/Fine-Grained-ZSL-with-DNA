import numpy as np
from scipy.special import gammaln, softmax  
from scipy.linalg import solve_triangular, solve, eigh, eig
from utils import data_loader, perf_calc_acc


class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model,self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']



        self.data = data_loader(datatpath, dataset, tuning)




    ### Claculating class mean and covariance priors
    def calculate_priors(self, xtrain, ytrain, model_v='unconstrained'):
        dim    = xtrain.shape[1]
        uy     = np.unique(ytrain)
        nc     = len(uy)
        if model_v=='constrained':
            class_means = np.zeros((nc,dim))
            for j in range(nc):
                idd = np.in1d(ytrain, uy[j])
                class_means[j] = np.mean(xtrain[idd], axis=0).T  
            mu_0        = np.mean(class_means, axis=0)
            Sigma_0     = 0
        else:
            Sigma_0, mu_0    = 0, 0
            for j in range(nc):
                idd = np.in1d(ytrain, uy[j])
                Sigma_0  += np.cov(xtrain[idd].T)
                mu_0 += np.mean(xtrain[idd], axis=0)
            Sigma_0     /= nc 
            mu_0        /= nc
        
        return mu_0, Sigma_0


    ### Calculating Posterior Predictive Distribution parameters ###
    def calculate_ppd_params(self, xtrain, ytrain, att_seen, att_unseen, unseenclasses, K, Psi, mu0, m, k0, k1):
        
        seenclasses = np.unique(ytrain)
        nc   = len(seenclasses) + len(unseenclasses)
        n, d = xtrain.shape
        
        Sig_s    = np.zeros((d,d,nc))
        Sigmas   = np.zeros((d,d,nc))
        mu_s     = np.zeros((nc, d))
        v_s      = np.zeros((nc, 1), dtype=np.int32)
        class_id = np.zeros((nc, 1))
        us_allclasses = set()
        
        #Start with the unseen classes
        uy            = unseenclasses
        ncl           = len(uy)
        cnt           = 0
        # Main for loop for  unseen classes params estimation
        for i in range(ncl):
        
            #Calculating Euclidean distance between the selected unseen class
            #attributes and all seen classes
            tmp       = att_unseen[i, np.newaxis]
            D         = cdist(att_seen, tmp)
            s_in = np.argsort(D.ravel())
        
            # Choose the K nearest neighbor for forming meta clusters
            classes   =  seenclasses[s_in[:K]]
            print('unseen class: %d' % (uy[i]))
            print('associated classes: ')
            print(classes)
    #         set_trace()
    #         ect       = 0
    #         while check_for_tie(us_allclasses, classes.T):
    #             classes[-1] = seenclasses[s_in[K+ect]]
    #             ect += 1

    #         us_allclasses.add(classes)    

            #Extract corresponding data
            idx       = np.in1d(ytrain, classes)
            Yi        = ytrain[idx]
            Xi        = xtrain[idx]
            uyi       = np.unique(Yi)

            # Initialize component sufficient statistics 
            ncpi      = len(uyi)
            xkl       = np.zeros((ncpi,d))      # Component means
            Skl       = np.zeros((d,d,ncpi))    # Component scatter matrices
            kap       = np.zeros((ncpi,1))      # model specific
            nkl       = np.zeros((ncpi,1))      # number of data points in the components
        
            # Calculate  sufficient statistics for each component in meta cluster
            for j in range(ncpi):
                idx        = np.in1d(Yi, uyi[j])
                nkl[j]     = np.sum(idx)
                kap[j]     = nkl[j]*k1/(nkl[j]+k1)
                Xij        = Xi[idx]
                xkl[j]     = np.mean(Xij, axis=0)
                Skl[:,:,j] = (nkl[j]-1)*np.cov(Xij.T)   
            
            # Model specific parameters
            sumkap       = np.sum(kap)
            kaps         = (sumkap+k0)*k1/(sumkap+k0+k1)
            sumSkl       = np.sum(Skl,axis=2)
            muk          = (np.sum(np.multiply(xkl, kap*np.ones((1,d))), axis=0)+k0*mu0)/(sumkap+k0)
            #set_trace()
            # Unseen classes' predictive cov, mean and dof
            v_s[cnt]         = np.sum(nkl)-ncpi+m-d+1
            class_id[cnt]   = uy[i]
            Sigmas[:,:,cnt] = Psi+sumSkl
            Sig_s[:,:,cnt]  = (Psi+sumSkl)/((kaps*v_s[cnt])/(kaps+1))
            mu_s[cnt]       = muk
            cnt             += 1  
        
        # The second part: same procedure for Seen classes
        uy              = seenclasses
        ncl             = len(uy)

        for i in range(ncl):
            idx         = np.in1d(ytrain, uy[i])
            Xi          = xtrain[idx]

            # The current selected component stats: # points, mean and scatter
            cur_n       = np.sum(idx)
            #print('seen class: %d, cur_n: %d' % (uy[i], cur_n))
            cur_S       = (cur_n-1)*np.cov(Xi.T)
            cur_mu      = np.mean(Xi, axis=0, keepdims=True)

            # Selected seen class attribute distance to all other seen class attr
            tmp        = att_seen[i, np.newaxis]
            D          = cdist(att_seen, tmp)
            s_in       = np.argsort(D.ravel())

            # neighborhood radius
            classes    =  seenclasses[s_in[1:K+1]]
            
            # !!! As shown in the PPD derivation of Supplementary material model
            # supports forming surrogate classes for seen classes as well but we
            # did not utilized local priors for seen classes in this work. We
            # just used data likelihood and global prior for seen class formation
            # as mentioned in the main text !!!
            # Thus nci is set to 0 instead of len(classes)
            nci         = 0 #length(classes)
            # To check whether there are more than one seen classes
            if nci>0:

                idx = np.in1d(ytrain, classes)
                Yi        = ytrain[idx]
                Xi        = xtrain[idx]
                uyi     = classes
                
                # data and initialization
                ncpi    = len(uyi)
                xkl     = np.zeros((ncpi, d))
                Skl     = np.zeros((d,d,ncpi))
                kap     = np.zeros((ncpi,1))
                nkl     = np.zeros((ncpi,1))

                # sufficient stats calculation
                for j in range(ncpi):
                    idx        = np.in1d(Yi, uyi[j])
                    nkl[j]    = np.sum(idx)
                    kap[j]     = nkl[j]*k1/(nkl[j]+k1)
                    Xij        = Xi[idx]              # Data points in component j and meta cluster i
                    xkl[j]     = np.mean(Xij, axis=0, keepsdim=True)
                    Skl[:,:,j] = (nkl[j]-1)*np.cov(Xij.T)   


                sumkap  = np.sum(kap)
                kaps    = (sumkap+k0)*k1/(sumkap+k0+k1)
                sumSkl  = np.sum(Skl, axis=2)
                muk     = (np.sum(np.multiply(xkl, kap*np.ones((1,d))), axis=0)+k0*mu0)/(sumkap+k0)
                vsc     = np.sum(nkl)-ncpi+m-d+1

                v_s[cnt]        = vsc+cur_n
                Smu             = ((cur_n*kaps)/(kaps+cur_n))*np.dot(cur_mu-muk, (cur_mu-muk).T)
                Sigmas[:,:,cnt] = Psi+sumSkl+cur_S+Smu         # Just need for exp of rebuttal, then delete
                Sig_s[:,:,cnt]  = (Psi+sumSkl+cur_S+Smu)/(((cur_n+kaps)*v_s[cnt])/(cur_n+kaps+1))
                mu_s[cnt]       = (cur_n*cur_mu+kaps*muk)/(cur_n+kaps)
                class_id[cnt]   = uy[i]
                cnt            += 1

                # The case where only data likelihood and global priors are used
                # and local priors are ignored. Thius is the case we used for seen
                # classes as mentioned in the paper
            else:
                v_s[cnt]        = cur_n+m-d+1
                mu_s[cnt]       = (cur_n*cur_mu+(k0*k1/(k0+k1))*mu0)/(cur_n+(k0*k1/(k0+k1)))
                Smu             = ((cur_n*(k0*k1/(k0+k1)))/((k0*k1/(k0+k1))+cur_n))*np.dot(cur_mu-mu0, (cur_mu-mu0).T)
                Sig_s[:,:,cnt]  = (Psi+cur_S+Smu)/(((cur_n+(k0*k1/(k0+k1)))*v_s[cnt])/(cur_n+(k0*k1/(k0+k1))+1))
                class_id[cnt]   = uy[i]
                cnt            +=1

        return Sig_s, mu_s, v_s, class_id, Sigmas 

    ### PPD calculation (Log-Likelihood of Student-t) ###
    def unconstrained_prediction(self, X, Sig_s, mu_s, v_s, class_id):
    
        # Initialization
        ncl, d   = mu_s.shape
        piconst  = (d/2)*np.log(np.pi)
        gl_pc      = gammaln(np.arange(0.5, np.max(v_s) + d + 0.5, 0.5)) 
        n          = X.shape[0] 
        lkh        = np.zeros((n,ncl))
        
        # Calculating log student-t likelihood
        for j in range(ncl):
            v        = X - mu_s[j]                           # Center the data
            chsig    = np.linalg.cholesky(Sig_s[:,:,j])      # Cholesky decomposition
            tpar     = gl_pc[v_s[j]+d-1] - (gl_pc[v_s[j]-1] + (d/2)*np.log(v_s[j])+piconst)-np.sum(np.log(chsig.diagonal())) # Stu-t lik part 1
            temp     = solve_triangular(chsig, v.T, overwrite_b=True, check_finite=False, lower=True).T # mrdivide(v,chsig)
            norm2    = np.einsum('ij,ij->i', temp, temp)     # faster than np.sum(temp**2)
            lkh[:,j] = tpar - 0.5*(v_s[j]+d)*np.log(1+(1/v_s[j])*norm2)

        bb           = np.argmax(lkh, axis=1)
        ypred        = class_id[bb]                 # To ensure labels are correctly assigned back to original ones

        return ypred, lkh


    ### Main function to combine all subfunction into one to finally build the classifier
    def bayesian_cls(self, x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, att, k_0=0.1, k_1=10, m=5*500, mu_0=0, s=1, scatter=0, K=2, pca_dim=0, tuning=False):
    us_classes  = np.unique(y_ts_us)
    s_classes   = np.unique(y_tr)
    
    # Attributes of seen and unseen classes
    att_unseen  = att[:, us_classes].T
    att_seen    = att[:, s_classes].T
    d0          = x_tr.shape[1]
    
    if pca_dim: 
        # Dimentionality reduction from PCA
        _, eig_vec = eigh(np.cov(x_tr.T))
        x_tr    = np.dot(x_tr, eig_vec[:, -pca_dim:])
        x_ts_s    = np.dot(x_ts_s, eig_vec[:, -pca_dim:])
        x_ts_us    = np.dot(x_ts_us, eig_vec[:, -pca_dim:])
        
        d0      = pca_dim

    print(d0)
    # Pre-calculation of Psi (prior covariance) from tuned scale s, and
    # scatter. The reason behind this if statement is that we dont want
    # repeat this precalculation in hypertuning since it is expensive
    # in time but we want to calculate this values with new data during
    # testing
    if self.tuning:
        Psi=(m-d0-1)*scatter/s
    else:
        [mu_0, scatter] = self.calculate_priors(x_tr, y_tr)
        Psi=(m-d0-1)*scatter/s
           
    # Class predictive cov, mean and DoF from unconstrained model
    Sig_s,mu_s,v_s,class_id,Sigmas    = self.calculate_ppd_params(x_tr, y_tr, att_seen,att_unseen,us_classes, K,Psi,mu_0,m,k_0,k_1)
    print('PPD derivation is Done!!')
    ####### Prediction phase #########
    ypred_us, prob_mat_us = self.unconstrained_prediction(x_ts_us, Sig_s, mu_s, v_s, class_id)
    print('Unseen class likelihoods estimeted')
    ypred_s, prob_mat_s    = self.unconstrained_prediction(x_ts_s, Sig_s, mu_s, v_s, class_id)
    print('Seen class likelihoods estimeted')
    acc_per_cls_s, acc_per_cls_us, gzsl_seen_acc, gzsl_unseen_acc, H = perf_calc_acc(y_ts_s, y_ts_us, ypred_s, ypred_us)

    return gzsl_seen_acc, gzsl_unseen_acc, H, acc_per_cls_s, acc_per_cls_us, prob_mat_s, prob_mat_us, class_id
