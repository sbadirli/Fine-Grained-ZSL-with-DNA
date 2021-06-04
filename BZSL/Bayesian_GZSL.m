    % Main function for running the algorithm
    % Inputs: (Musts) 
    %   training data:              x_tr, y_tr
    %   Test data (seen):           x_ts_s, y_ts_s    
    %   Test data (unseen):         x_ts_us, y_ts_us 
    %   Attributes:                 att
    %   Version of 'Model':         'constrained', 'unconstrained' |                             
    % Optional inputs:
    %   Hyperparameters:            ['kappa_0','kappa_1','cov_shape','invg_shape','invg_scale','prior_mean','prior_covscale']
    %                               'prior_mean','prior_covscale' as mu_0 & s
    %                               'cov_shape' stands for m in the paper
    %                               and 'invg_shape','invg_scale' for a_0 &
    %                               b_0 respectively.
    %   Number of iters:            'iter', accepts integer
    %   # components for PCA :      'pca',  accepts positive integer. 
    %                               0 means no need for PCA.
    %   Tuning option:              'tuning', true or false.
    %   # of Nearest neighbor:      'num_neighbor', K used in finding the nearest neighbors among seen classes of unseen class 
    % Outputs:
    %   Accuracy for seen classes:   gzsl_seen_acc
    %   Accuracy for unseen classes: gzsl_unseen_acc  
    %   Harmonic mean:               H
    
    

function [gzsl_seen_acc, gzsl_unseen_acc, H, acc_per_sclass, acc_per_usclass, prob_mat_s, prob_mat_us, class_id] = Bayesian_GZSL(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, att, varargin)
    
    % Seen and unseen classes
    us_classes  = unique(y_ts_us);
    s_classes   = unique(y_tr);
    
    % Attributes of seen and unseen classes
    att_unseen  = att(:,us_classes)';
    att_seen    = att(:,s_classes)';
    d0          = size(x_tr, 2);

    % Parsing passed parameters and  hyperparameters from tuning
    params      = varargin;
    [k_0,k_1, m, a_0, b_0, mu_0, s, scatter, K, num_iter, pca_dim, Model, tuning] = hyperparameter_setting(x_tr, params);
    
    % num_iter for repeating the procedure several times to eleminate
    % randomness
    % You may change the # features to use by changing d0
    if pca_dim 
        % Dimentionality reduction from PCA
        C       = cov(x_tr);
        [vv, ~] = eig(C);
        x_tr    = x_tr*vv(:,end-pca_dim+1:end);
        x_ts_s  = x_ts_s*vv(:,end-pca_dim+1:end);
        x_ts_us = x_ts_us*vv(:,end-pca_dim+1:end);
        
        d0      = pca_dim;
    end
        
    % Mixing feature positions
    if tuning
        for i=1:num_iter
        fin{i}  = 1:d0;
        end
    else 
            for i=1:num_iter
        tmp     = randperm(d0);
        fin{i}  = tmp(1:d0);
        fin{i}  = 1:d0;
            end
    end
    
        

    % Main for loop for the calculations
    for iter=1:numel(fin)
        
        % training data
        xn         = x_tr(:,fin{iter});
        yn         = y_tr;
        % Test data from seen and unseen classes (GZSL)
        xt_unseen  = x_ts_us(:,fin{iter});
        xt_seen    = x_ts_s(:,fin{iter});
        
        % Pre-calculation of Psi (prior covariance) from tuned scale s, and
        % scatter. The reason behind this if statement is that we dont want
        % repeat this precalculation in hypertuning since it is expensive
        % in time but we want to calculate this values with new data during
        % testing
        if tuning
            Psi=(m-d0-1)*scatter/s;
        else
            [mu_0, scatter] = calculate_priors(xn, yn, 'Model', Model);
            Psi=(m-d0-1)*scatter/s;
        end

        if strcmp(Model,'constrained')

            % Component predictive cov, mean and DoF from constrained model
            [Sig_s,mu_s,v_s,class_id] = constrained_estimation(xn,yn,att_seen,att_unseen,us_classes, K,mu_0,k_0,k_1,a_0,b_0);
            
            % Predicting labels for the test data from unseen and seen classes 
            % utilizing cov, mean and dof from the model. Max likelihood with stu-t 
            [ypred_unseen(:,iter), prob_mat_us] = constrained_predicting(xt_unseen, Sig_s, mu_s, v_s, class_id);
            [ypred_seen(:,iter), prob_mat_s]    = constrained_predicting(xt_seen, Sig_s, mu_s, v_s, class_id);
        else
           
            % Class predictive cov, mean and DoF from unconstrained model
            [Sig_s,mu_s,v_s,class_id,Sigmas]    = unconstrained_estimation(xn,yn,att_seen,att_unseen,us_classes, K,Psi,mu_0,m,k_0,k_1);
            
            % Prediction phase
            [ypred_unseen(:,iter), prob_mat_us] = unconstrained_prediction(xt_unseen, Sig_s, mu_s, v_s, class_id);
            [ypred_seen(:,iter), prob_mat_s]    = unconstrained_prediction(xt_seen, Sig_s, mu_s, v_s, class_id);
        end
        

    end
        
    % Mode of these iterations to alleviate the effect of r.v. in
    % constrained model
    % In our experiments we just used 1 iteration as in unconstrained model
    % we do not have random effect
    ypred1          = mode(ypred_unseen,2);
    classes         = unique(y_ts_us);
    nclass          = length(classes);
    acc_per_usclass   = zeros(nclass, 1);

    % Top-1 Accuracy per unseen class
    for i=1:nclass
        idx              = find(y_ts_us==classes(i));
        acc_per_usclass(i) = sum(y_ts_us(idx) == ypred1(idx)) / length(idx);
    end

    % Generalized zero-shot learning unseen class accuracy -- average
    gzsl_unseen_acc = mean(acc_per_usclass);


    % Accuracy calculation for seen classes
    ypred1          = mode(ypred_seen,2);
    classes         = unique(y_ts_s);
    nclass          = length(classes);
    acc_per_sclass   = zeros(nclass, 1);
    for i=1:nclass
        idx = find(y_ts_s==classes(i));
        acc_per_sclass(i) = sum(y_ts_s(idx) == ypred1(idx)) / length(idx);
    end

    gzsl_seen_acc   = mean(acc_per_sclass);

    % Harmonic mean for seen and unseen classes acc. from Y. Xian paper
    H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc);   

end