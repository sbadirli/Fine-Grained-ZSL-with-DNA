% This function splits the training data into Train and Test data for the
% purpose of hyper-parameter tuning

function [xtrain, ytrain, xtest_unseen, ytest_unseen, xtest_seen, ytest_seen] = tuning_split(features, train_loc, val_loc, labels, fname2)

% Transpose to get a usual shape of the feature matrix
features     = features';

% Training data and labels
X            = features(train_loc ,:);
Y            = labels(train_loc); 

% Creating new training and test data from training data without involving
% real test data. 10% of the training data is left for testing
if ~exist('train')
    disp('Splitting data for hyperparameter tuning ...')
    [train,test] = crossvalind('holdout',Y,0.2); 
    save(fname2,'-append','train','test');
end

xtrain       = X(train,:); 
ytrain       = Y(train);

% Test data and labels, Generalized ZSL setting: Seen and Unseen
xtest_seen   = X(test,:); 
ytest_seen   = Y(test);

xtest_unseen = features(val_loc,:); 
ytest_unseen = labels(val_loc);

end




    





