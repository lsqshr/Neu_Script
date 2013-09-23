load super331;
d = data;
data.CONVEXITY = [d.CONVEXITY(1:77,:);d.CONVEXITY(247:end,:)];
data.SOLIDITY  = [d.SOLIDITY(1:77,:);d.SOLIDITY(247:end,:)];
data.VOLUME    = [d.VOLUME(1:77,:);d.VOLUME(247:end,:)];
data.labels    = [d.labels(1:77);d.labels(247:end)];
data.labels(78:end) = 2;

save('NCAD331.mat','data');

data.CONVEXITY = d.CONVEXITY(1:246,:);
data.SOLIDITY  = d.SOLIDITY(1:246,:);
data.VOLUME    = d.VOLUME(1:246,:);
data.labels    = d.labels(1:246);
data.labels(78:end) = 2;

save('NCMCI331.mat','data');