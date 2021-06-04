% This function loads the hyperparameter set from CV for a scpecified dataset.
% Please refer to the tuning  range and other tuning details in the Supp.
% Materials.

function [att, K, k_0, k_1, m, s, a_0 , b_0] = load_tuned_params(fname, dataset, side_info)
    
    if strcmp(dataset, 'INSECT') & ~strcmp(side_info, 'dna')
        fprintf('Invalid side information source! There is only one side information source for INSECT dataset: "dna" \n')
        fprintf('Model will continue using DNA as side information\n')
    end
    load(fname);
    dataset = upper(dataset);
    dim  = 500;
    
    INSECT  = [0.1, 10, 5*dim, 10, 3];
    if strcmp(dataset, 'CUB_DNA')
        if strcmp(side_info, 'visual')
            CUB_DNA  = [1, 25, 500*dim, 10, 3];
        elseif strcmp(side_info, 'w2v')
            CUB_DNA  = [0.1, 25, 5*dim, 5, 2];
            att = att_w2v;
        else
            CUB_DNA  = [0.1, 25, 25*dim, 5, 3];
            att = att_exp;
        end
    end
    
    eval(['data = ', dataset,';']);
    data = num2cell(data);
    [k_0, k_1, m, s, K] = deal(data{:});
end