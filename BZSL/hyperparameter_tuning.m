% This function performs hyper-parameter tuning utilizing Bayesian ZSL and
% extract the parameter tuple with the highest Harmonic mean. The model
% version, constrained or unconstrained, must be specified.
% Inputs:
%   Data  -- Training and test + attributes 
%   Model versions -- 'constrained', 'unconstrained'| Ex:  ['Model', 'constrained']
%
% Optional: (if the model version is specified as unconstrained)
%   # components for PCA -- dim  | Ex: ['pca', 300]     


function [k0, k1, mm, a0, b0, mu_0, s, K] = hyperparameter_tuning(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen, att, varargin)
    
    % Default # features for PCA id Unconstrained model selected
    dim      = 500;

    if nargin<2
        fprintf('The model version is missing!! Please insert version name: constrained or unconstrained \n');
    elseif nargin==4
        dim  = varargin{4};
    end
   
    % Tuning range for the parameters
    k0_range = [0.1 1]; 
    k1_range = [10 25];
    a0_range = [1 10 100];
    b0_range = a0_range;
    s_range  = [1 5 10];
    K_range  = [1 2 3];
    
    % Initialization
    s        = 0; a0 = 20; b0 = 20; mm = dim + 2;
    
    bestH    = 0; best_acc_s = 0; best_acc_us = 0;
    
    if strcmp(varargin{2}, 'unconstrained')
        
        % PCA for dimentionality reduction
        fprintf('Applying PCA to reduce the dimension...\n')
        C            = cov(xtrain);
        [vv, ~]      = eig(C);
        xtrain       = xtrain*vv(:,end-dim+1:end);
        xtest_seen   = xtest_seen*vv(:,end-dim+1:end);
        xtest_unseen = xtest_unseen*vv(:,end-dim+1:end);
        
        
        
        % Precalculation of class means and scatter matrices for unconstrained model
        [mu_0, scatter] = calculate_priors(xtrain, ytrain, 'Model', 'unconstrained');
        
        % m range is determined by the dim of data
        m_range  = [5*dim 25*dim 100*dim 500*dim];
        fprintf('Tuning is getting started...\n')
        for kk=K_range
            for k_0=k0_range
                for k_1=k1_range
                    for m=m_range
                        for ss=s_range
                            %Psi=(m-dim-1)*scatter/s;
                            tic
                            [acc_s, acc_us, H, s_cls_acc, us_cls_acc, pb_s, pb_us, class_id] = Bayesian_GZSL(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen, att,'Model', 'unconstrained','tuning',true,'num_neighbor', kk,...
                                                'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'prior_mean', mu_0,'prior_covscale', ss,'scatter', scatter, 'pca', 0);

                            % Print out when there is an improvement in
                            % Harmonic mean
                            if H>bestH  %|| acc_s>best_acc_s || acc_us>best_acc_us
                                bestH = H;% best_acc_s = acc_s; best_acc_us = acc_us;
                                k0 = k_0; k1 = k_1; mm = m; s  = ss; K = kk;
                                disp(['GZSL unseen: averaged per-class accuracy=' num2str(acc_us) ]);
                                disp(['GZSL seen: averaged per-class accuracy=' num2str(acc_s) ]);
                                disp(['GZSL: H=' num2str(H)]);
                                disp(['K=' num2str(kk)]);
                                disp(['k0=' num2str(k_0)]);
                                disp(['k1=' num2str(k_1)]);
                                disp(['m=' num2str(m)]);
                                disp(['s=' num2str(s)]);

                            end
                            toc
                        end
                    end
                end
            end
        end
    else
        
        d           = size(xtrain, 2);
        %m_range     = [10*d 50*d 500*d];
        % Precalculation of class means and scatter matrices for unconstrained model
        fprintf('Calculating priors for the Constrained model ...');
        tic
        [mu_0, ~] = calculate_priors(xtrain, ytrain, 'Model', 'constrained');
        toc
        fprintf('Tuning is getting started...\n')
        for kk=K_range
            for k_0=k0_range
                for k_1=k1_range
                    for a_0=a0_range
                        for b_0=b0_range

                            tic
                            [acc_s,acc_us,H] = Bayesian_ZSL(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen, att,...
                                                'Model', 'constrained','tuning', true,'num_neighbor', kk, 'kappa_0', k_0, 'kappa_1', k_1,...
                                                'prior_mean', mu_0,'invg_shape', a_0, 'invg_scale', b_0);
                            % Print out when there is an improvement in
                            % Harmonic mean
                            if  H>bestH  %|| acc_us>best_acc_us % || acc_s>best_acc_s
                                bestH = H; % best_acc_us = acc_us; % best_acc_s = acc_s;
                                k0 = k_0; k1 = k_1; a0 = a_0; b0 = b_0; K = kk;
                                disp(['GZSL unseen: averaged per-class accuracy=' num2str(acc_us) ]);
                                disp(['GZSL seen: averaged per-class accuracy=' num2str(acc_s) ]);
                                disp(['GZSL: H=' num2str(H)]);
                                disp(['K=' num2str(kk)]);
                                disp(['k0=' num2str(k_0)]);
                                disp(['k1=' num2str(k_1)]);
                                disp(['a0=' num2str(a_0)]);
                                disp(['b0=' num2str(b_0)]);
                            end
                            toc

                        end 
                    end
                end
            end
        end

    end