load results;
nresults = results{10000};
for i = 166 : nresults
    result = results{i,1} ;
    
    disp([result{8} result{3} result{2} ]);
    disp('-------');
    disp(result{11});
    disp('*********');
end