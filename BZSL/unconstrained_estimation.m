 % This function is the one used for hyper-parameter tuning in Unconstrained
% model
% Inputs: 
%   training data:      X, Y
%   Attributes:         for both type of classes, att_seen & att_unseen
%   Unseen class names: unseenclasses
%   Hyperparameters:    
%                       mu0     -- initial mean
%                       m       -- 
%                       k0      -- kappa_0
%                       k1      -- kappa_1
%                       Psi     -- Iinitial covariance matrix
%                       K       -- The # of nearest neighbors of the unseen
%                                  class among seen classes
%   Version of algo:    Unconstrainted 
%
% Outputs:
%   Class predictive covariances:  Sig_s
%   Class predictive means:        mu_s
%   Class predictive DoF:          v_s
%   Class ids:                     class_id

function [Sig_s,mu_s,v_s,class_id,Sigmas]=unconstrained_estimation(X,Y,att_seen,att_unseen,unseenclasses,K,Psi,mu0,m,k0,k1)

% data stats
seenclasses   = unique(Y);
nc            = length(seenclasses)+length(unseenclasses);
[n, d]        = size(X);

% Initialize output params: derive for each class predictive cov, mean and dof 
Sig_s         = zeros(d,d,nc);
Sigmas         = zeros(d,d,nc);
mu_s          = zeros(nc,d);
v_s           = zeros(nc,1);

% Start with the unseen classes
uy            = unseenclasses;
ncl           = length(uy);
cnt           = 1; 

us_allclasses = {};
% Main for loop for  unseen classes params estimation
for i=1:ncl
    
    % Calculating Euclidean distance between the selected unseen class
    % attributes and all seen classes
    tmp       = att_unseen(i,:);
    D         = pdist2(att_seen,tmp);
    [~, s_in] = sort(D,'ascend');
    in        = false(n,1);
    
    % Choose the K nearest neighbor for forming meta clusters
    classes   =  seenclasses(s_in(1:K));
    ect       = 1;
    while check_for_tie(us_allclasses, classes')
        classes(end) = seenclasses(s_in(K+ect));
        ect = ect+1;
    end
            
    us_allclasses{i} = classes';    
    % Marking the meta cluster classes
    nci       = length(classes);
    for j=1:nci
        in(Y==classes(j))=1;
    end
    
    % Extract corresponding data
    Yi        = Y(in);
    Xi        = X(in,:);
    uyi       = unique(Yi);
    
    % Initialize component sufficient statistics 
    ncpi      = length(uyi);
    xkl       = zeros(ncpi,d);      % Component means
    Skl       = zeros(d,d,ncpi);    % Component scatter matrices
    kap       = zeros(ncpi,1);      % model specific
    nkl       = zeros(ncpi,1);      % # data points in the components
    
    % Calculate  sufficient statistics for each component in meta cluster
    for j=1:ncpi
        in         = Yi==uyi(j);
        nkl(j)     = sum(in);
        kap(j)     = nkl(j)*k1/(nkl(j)+k1);
        Xij        = Xi(in,:);
        xkl(j,:)   = mean(Xij,1);
        Skl(:,:,j) = (nkl(j)-1)*cov(Xij);   
    end
    
    % Model specific parameters
    sumkap       = sum(kap);
    kaps         = (sumkap+k0)*k1/(sumkap+k0+k1);
    sumSkl       = sum(Skl,3);                                          % sum of scatters
    muk          = (sum(xkl.*(kap*ones(1,d)),1)+k0*mu0)/(sum(kap)+k0);  % meta cluster mean
    
    % Unseen classes' predictive cov, mean and dof
    vsc             = sum(nkl)-ncpi+m-d+1;
    class_id(cnt,:) = uy(i);
    v_s(cnt)        = vsc;
    Sigmas(:,:,cnt) = Psi+sumSkl;
    Sig_s(:,:,cnt)  = (Psi+sumSkl)/(((kaps)*v_s(cnt))/(kaps+1));
    mu_s(cnt,:)     = muk;
    cnt             = cnt+1;  
end

% The second part: same procedure for Seen classes
uy              = seenclasses;
ncl             = length(uy);

for i=1:ncl
    in          = Y==uy(i);
    Xi          = X(in,:);
    
    % The current selected component stats: # points, mean and scatter
    cur_n       = sum(in);
    cur_S       = (cur_n-1)*cov(Xi);
    cur_mu      = mean(Xi,1);
    
    % Selected seen class attribute distance to all other seen class attr
    tmp         = att_seen(i,:);     
    D           = pdist2(att_seen,tmp);
    [~, s_i]    = sort(D,'ascend');
    in          = false(n,1);
    
    % neighborhood radius
    classes     = seenclasses(s_i(2:(K+1)));
    
    %%% !!! As shown in the PPD derivation of Supplementary material model
    %%% supports forming surrogate classes for seen classes as well but we
    %%% did not utilized local priors for seen classes in this work. We
    %%% just used data likelihood and global prior for seen class formation
    %%% as mentioned in the main text !!!
    %%% Thus nci is set to 0 instead of length(classes);
    nci         = 0; %length(classes);
    
    % To check whether there are more than one seen classes
    if nci>0
        
        for j=1:nci
            in(Y==classes(j))=1;
        end
        
        % data and initialization
        Yi      = Y(in);
        Xi      = X(in,:);
        uyi     = unique(Yi);
        ncpi    = length(uyi);
        xkl     = zeros(ncpi,d);
        Skl     = zeros(d,d,ncpi);
        kap     = zeros(ncpi,1);
        nkl     = zeros(ncpi,1);
        
        % sufficient stats calculation
        for j=1:ncpi
            in         = Yi==uyi(j);
            nkl(j)     = sum(in);
            kap(j)     = nkl(j)*k1/(nkl(j)+k1);
            Xij        = Xi(in,:);              % Data points in component j and meta cluster i
            xkl(j,:)   = mean(Xij,1);
            Skl(:,:,j) = (nkl(j)-1)*cov(Xij);   
        end
        
        
        sumkap  = sum(kap);
        kaps    = (sumkap+k0)*k1/(sumkap+k0+k1);
        sumSkl  = sum(Skl,3);
        muk     = (sum(xkl.*(kap*ones(1,d)),1)+k0*mu0)/(sum(kap)+k0);
        vsc     = sum(nkl)-ncpi+m-d+1;

        v_s(cnt)        = vsc+cur_n;
        Smu             = ((cur_n*kaps)/(kaps+cur_n))*((cur_mu-muk)*(cur_mu-muk)');
        Sigmas(:,:,cnt) = Psi+sumSkl+cur_S+Smu;   % Just need for exp of rebuttal, then delete
        Sig_s(:,:,cnt)  = (Psi+sumSkl+cur_S+Smu)/(((cur_n+kaps)*v_s(cnt))/(cur_n+kaps+1));
        mu_s(cnt,:)     = (cur_n*cur_mu+kaps*muk)/(cur_n+kaps);
        class_id(cnt,1) = uy(i);
        cnt             = cnt+1;
        
        % The case where only data likelihood and global priors are used
        % and local priors are ignored. Thius is the case we used for seen
        % classes as mentioned in the paper
        else
            v_s(cnt)        = cur_n+m-d+1;
            mu_s(cnt,:)     = (cur_n*cur_mu+(k0*k1/(k0+k1))*mu0)/(cur_n+(k0*k1/(k0+k1)));
            Smu             = ((cur_n*(k0*k1/(k0+k1)))/((k0*k1/(k0+k1))+cur_n))*((cur_mu-mu0)*(cur_mu-mu0)');
            Sig_s(:,:,cnt)  = (Psi+cur_S+Smu)/(((cur_n+(k0*k1/(k0+k1)))*v_s(cnt))/(cur_n+(k0*k1/(k0+k1))+1));
            class_id(cnt,1) = uy(i);
            cnt             = cnt+1;
    end

    
end

 