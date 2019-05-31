function [new_data, rem_indexes] = myEntropy(dataset, num)
% Based on similarity measure and De Luca and Termini’s definition for fuzzy measure
% it calculates importance of features and removes the least important features
% INPUT:
%  dataset - (M+1)xN dataset matrix, where N - number of samples, M - length of each
% sample (also size of input layer) + one row of classes
%  num - number of features to remove
%  
% OUTPUT:
%  new_data - (M-num)xN matrix of features without removed features
%  rem_indexes - an array of removed feature indexes (indexes of initial
%  dataset)

data2 = dataset(1:end-1,:);
data = (minmaxnorm(data2',0,1))';
classes = dataset(end,:);
[M,N] = size(data);

c0 = data(:,classes==0);
c1 = data(:,classes==1);
[c1M,c1N] = size(c0);
[c2M,c2N] = size(c1);
p=1;
for i=1:c1M
    m1(i) = mean(c0(i,:));
end
for i=1:c2M
    m2(i) = mean(c1(i,:));
end

for i=1:c1N
    s1(i,:)=simi(c0(:,i)',m1,p);
end
for i=1:c2N
    s2(i,:)=simi(c1(:,i)',m1,p);
end
S=[s1;s2];
for i=1:M
    ev(i)=entropyDeLuca2(S(:,i));
end

av = mean(ev);
minev = min(ev);
maxev = max(ev);
ev2 = sort(ev);
for i=0:23
    maximal5(i+1)=ev2(end-i);
end

indexes = [];


for i = 1:num
    
    indexes = [indexes, find(ismember(ev, maximal5(i)))];
end

rem_indexes = unique(indexes);
new_indexes = setdiff(1:M,rem_indexes);
new_data = data(new_indexes,:);
end
