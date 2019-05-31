function [accuracy, test_result] = similarity_classifier(train_data, train_class, test_data, test_class, p, m)
%  INPUT:
%  train_data - MxN matrix, where N - number of samples, M - length of each sample 
%  train_class - KxN array of classes, where K - number of classes for each sample
%  test_data - MxN_2 matrix for testing, where N_2 - number of test samples
%  test_class - KxN_2 array of classes, where K - number of classes for each sample
%  p and m - degrees for similarity measures
%
%  OUTPUT:
%  test_result - KxN_2 array of classes for test_data
%  accuracy - accuracy on predicted test set

[tN,tM] = size(test_data);
class_values = unique(train_class);
num_of_classes = size(class_values,2);
c = {};
w = ones(num_of_classes,1);
for i=1:num_of_classes
    c{i} = train_data(:,train_class==class_values(i));
    [cN,cM] = size(c{i});
    for j=1:cN
        v(j) = (mean((c{i}(j,:)).^p)).^(1/p);
    end
    for j=1:tM
        S(i,j) = (sum( (1-abs( v.^p - test_data(:,j)'.^p)).^(m/p) )/tN).^(1/m);
    end
end

[test_val,test_result] = max(S);
test_result=test_result-1;
accuracy = 1 - sum(abs(test_class-test_result))/tM;
end

