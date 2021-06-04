% This function check for tie in the unseen meta-class formation. If the
% selected K seen classes appear in one of the previous meta-0classes it
% will return boolean True.
function flag = check_for_tie(all_classes, curr_classes)
    len = numel(all_classes);
    flag = 0;
    for i=1:len
        if isempty(setdiff(all_classes{i}, curr_classes)) && length(curr_classes)==length(all_classes{i})
            flag = 1;
            break;
        end
    end    
end