function [M, labels] = loaddata(path, features)
	addpath ..;
	load(path);

	M = [];
    
    for i = 1 : length(features)
        f = data.(features{i});
        switch(features{i})
            case {'CURVATURE', 'CONVEXITY', 'LGI'}
                f = f ./ max(f(:));
            case 'ShapeIndex'
                f = f - min(f(:));
                f = f ./ max(f(:));                                
        end
        M = [M f];
    end

    M = M';
	labels = data.labels;
end
