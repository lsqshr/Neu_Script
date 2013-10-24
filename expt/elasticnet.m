function elasticnet(data, labels)
    [b,fitinfo]  = lasso(data',labels,'CV',10,'Alpha',0.8);
    elasticresult.b       = b;
    elasticresult.fitinfo = fitinfo;
    save('elastic.mat','elasticresult');
end