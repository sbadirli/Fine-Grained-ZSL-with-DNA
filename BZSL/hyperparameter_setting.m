% This function parses the input variables into parameters
function [k_0,k_1, m, a_0, b_0, mu_0, s, scatter, K, num_iter, pca_dim, Model, tuning] = hyperparameter_setting(Data, hy)
    
    p               = inputParser;      % input parser struct
    dim             = 500;
    % Default values for hyperparameters
    default_model   = 'constrained';
    default_k0      = 0.01; 
    default_k1      = 10;
    default_m       = 10*2048; 
    default_a0      = 20;
    default_b0      = 20;
    default_mu0     = 0; 
    default_s       = 3;   
    default_iter    = 1;
    default_pca     = 0;
    default_scatter = 0;
    default_tuning  = false;
    default_K       = 2;
 
    expected_models = {'constrained', 'unconstrained'};   % Expected model versions
    
    addOptional(p,'Model', default_model, @(x) any(validatestring(x,expected_models)));
    addOptional(p,'kappa_0',default_k0,@isnumeric);
    addOptional(p,'kappa_1',default_k1,@isnumeric);
    addOptional(p,'cov_shape',default_m,@isnumeric);
    addOptional(p,'invg_shape',default_a0,@isnumeric);
    addOptional(p,'invg_scale',default_b0,@isnumeric);
    addOptional(p,'prior_mean',default_mu0,@isnumeric);
    addOptional(p,'prior_covscale',default_s,@isnumeric);
    addOptional(p,'iter',default_iter,@isnumeric);
    addOptional(p,'pca', default_pca,@isnumeric);
    addOptional(p, 'scatter', default_scatter,@isnumeric);
    addOptional(p, 'num_neighbor', default_K,@isnumeric);
    addOptional(p, 'tuning', default_tuning, @islogical)
    
    parse(p, hy{:});
    
    if strcmp(p.Results.Model, 'constrained')
        dim  = size(Data, 2);   
    end
        

    k_0      = p.Results.kappa_0;
    k_1      = p.Results.kappa_1;
    m        = p.Results.cov_shape;
    a_0      = p.Results.invg_shape;
    b_0      = p.Results.invg_scale;
    num_iter = p.Results.iter;
    pca_dim  = p.Results.pca;
    mu_0     = p.Results.prior_mean;
    s        = p.Results.prior_covscale;
    scatter  = p.Results.scatter;
    Model    = p.Results.Model;
    tuning   = p.Results.tuning;
    K        = p.Results.num_neighbor;
    
    if p.Results.prior_mean == 0
        mu_0 = zeros(1, dim);
    elseif p.Results.scatter == 0
        scatter  = eye(dim);
    end
    
        
