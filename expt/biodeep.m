clear;
lhiddenlist = {[600 500] [700 500] [700 600] [800 700]};
%lhiddenlist = {[2] [2] [2] [2]};
for i = 1 : numel(lhiddenlist)
    godeep(lhiddenlist{i},'bio',0);
end
