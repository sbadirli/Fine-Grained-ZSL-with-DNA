
% Bayesian Zero Shot Learning
%%%%%%%%%%%%%%%%   DEMO    %%%%%%%%%%%%%%%%%%%%
% This fuction demostrates the usage of BZSL on the datasets in (Generalized)
% Zero-shot learning domain: CUB, aPY, SUN, AWA & FLO
% The flow of the Demo is as follows,
% 1.    Load the one of the abovementioned data, they have specific train and
% test split nad attached attributes for each class. We have utilized the
% train, test and validation split provided in the datasets itself.
%
% 2.    Partitioning the training data into training and test data for the
% purpose of hyperparamter tuning. tuning_split.m script splits the training
% data into training and test (2 seperate sets: seen and unseen classes).  
% All these datasets have their ZSL specified partitioning and we followed  
% the exact same splitting fashion as in the literature.
%
% 3.    Hyperparameter tuning for the parameters of the BZSL. The matlab
% scritp hyperparameter_tuning.m implements this tuning. The must inputs
% for the tuning is the data split from tuning_split, attriburtes and Model
% version of the BZSL, constrained or unconstrained. If the unconstrained
% model is chosen you can specify the # principal components in the PCA which 
% used as dimensionality reduction technique in unconstrained model. EX: 
%   hyperparameter_tuning(Data split...,'Model', 'unconstrained', 'pca', 500);
% The script returns the desired hyperparameters  required for each model.
% For more details about hyperparameters please refer to the paper.
%
% 4. After getting the parameters, we used the train and test split
% provided within  datasets. We made use of Generalized ZSL setting for the
% split.
%
% 5. Lastly the BZSL model is run. The must inputs for the BZSL is data
% split from step 4, attributes, and the hyperparameters from tuning. Most
% of the hyper-parameters are the same for both models yet there are some
% differences. Please refer to the paper for the full list of
% hyperparameters of each model version. Since random number generation
% involved in the constrained model you could define number of iterations
% in this model to alleviate the randomness on the H score MOreover, there 
% is a tuning option for the BZSL where it is faster in this option if tuning 
% is being performed; ['tuning', true]  Ex: (constrained)
%     Bayesian_ZSL(Data split...,'Model', 'constrained', 'prior_mean', mu_0,...
%     'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'invg_shape', a_0, 'invg_scale', b_0, 'iter', 5);
%
% Ex: (unconstrained)
%     Bayesian_ZSL(Data split...,'Model','unconstrained', 'tuning', true, 'prior_mean', mu_0, 
%     'prior_cov', psi, 'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'pca', 500);

clear all;
%clc;

datapath = '..\data\';
dataset = 'INSECT'; %  {'INSECT', 'CUB_DNA'}

fname1=[datapath, dataset, '\res101.mat'];
fname2=[datapath, dataset, '\att_splits.mat'];
load(fname1)
load(fname2)

% Splitting the training data into training and test data for the use in
% hyper-parameter tuning
% [xtrain, ytrain, xtest_unseen, ytest_unseen, xtest_seen, ytest_seen] = tuning_split(features, train_loc, val_loc, labels, fname2);

% Tuning process and optimal parameters from CV
% [k_0, k_1, m, a_0, b_0, mu_0, s, K] = hyperparameter_tuning(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen,att,'Model', 'unconstrained', 'pca', 500);

% Splitting the data into train and test
% tr -- train, ts -- test, s -- seen, us -- unseen, x -- data matrix, 
% y -- labels 

% If you want to just reproduce the results from the paper use the following
% hyperparameter settings from Supplementary materiall for a specific dataset

% We have a constrained version of the model, but we did not use that
% version for any of the experiments. Thus please do not change
% model_version
model_version = 'unconstrained'; % DO NOT change, please!

% You may specify side information source below from this set: {'visual', 'w2v', 'dna'}.
% For INSECT we only have 'dna' but for CUB data we have all 3 sources.
side_info = 'dna'; % one of {'visual', 'w2v', 'dna'}
pca_dim = 500;

[att, K, k_0, k_1, m, s] = load_tuned_params(fname2, dataset, side_info);

% Splitting the data into train and test in accordance with the common proposed
% split used in ZSL benchmarking
[x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s] = split_data(features, trainval_loc, test_unseen_loc, test_seen_loc, labels);
tic
[gzsl_seen_acc, gzsl_unseen_acc, H, s_cls_acc, us_cls_acc, pb_s, pb_us, class_id] = Bayesian_GZSL(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, att,'Model', model_version,'num_neighbor', K,...
                                                                       'kappa_0', k_0, 'kappa_1', k_1,'cov_shape', m,'prior_covscale',s, 'pca', pca_dim);
fprintf('Results from k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d\n', k_0, k_1, m, s, K);
fprintf('BZSL results on %s dataset: Seen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%% \n', dataset, gzsl_seen_acc*100, gzsl_unseen_acc*100, H*100);
toc                        