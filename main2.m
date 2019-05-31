clc
clear all

name = 'bank-full.csv';
dataset = preprocess_data(name);

E = input("\nEnter 1 to implement Entropy measure or 0 for classifying without it ")


if(E)
    [data, remInd] = removeFeatures(dataset, 5);
else
    data = dataset(1:end-1, :);
end

classes = dataset(end,:);
[M,N] = size(data);
p = randperm(N);
trainN = fix(N*0.8);
testN = fix(N*0.2);
to_train = p(1:trainN);
to_test = p(trainN+1:trainN+1+testN);
train_data = data(:, to_train); % "train"
test_data =  data(:, to_test); % "test"
train_class = classes(:, to_train);
test_class  = classes(:, to_test);

[test_cls, w, err, ep] = multilayer_perceptron(train_data, train_class, test_data, [M fix(M/3) 4 1], [80 20], 0.1, 0.001, 100);
got_classes = zeros(size(test_cls));
got_classes(test_cls>0.5)=1;
misclassified = sum(got_classes~=test_class);
accuracy = sum(got_classes==test_class)/length(test_class);
fprintf('Number of misclassified test samples: %04d out of %04d\nAccuracy: %10.6f\n', ...
    misclassified, size(got_classes,2), accuracy);

[sim_accuracy, sim_classes] = similarity_classifier(train_data, train_class, test_data, test_class, 2, 2);
misclassified2 = sum(sim_classes~=test_class);
fprintf('\nAccuracy for similarity classifier: %10.6f\nNumber of misclassified test samples: %04d out of %04d\n', ...
 sim_accuracy, misclassified2, size(sim_classes,2));
