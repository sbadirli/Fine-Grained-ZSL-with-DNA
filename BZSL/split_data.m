% This function splits the data into 
%     Train, Test: Seen and Unseen

function [xtrain, ytrain, xtest_unseen, ytest_unseen, xtest_seen, ytest_seen] = split_data(features, trainval_loc, test_unseen_loc, test_seen_loc, labels)

% Transpose to get a usual shape of the feature matrix
features  = features';

% Training data and labels
xtrain=features(trainval_loc ,:);
ytrain=labels(trainval_loc);

% Test data and labels, Generalized ZSL setting: Seen and Unseen
xtest_unseen=features(test_unseen_loc,:);
ytest_unseen=labels(test_unseen_loc);
xtest_seen=features(test_seen_loc,:);
ytest_seen=labels(test_seen_loc);

end