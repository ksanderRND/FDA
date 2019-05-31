function [test_class, weights, full_error, current_epoch] = multilayer_perceptron(train_data, trainclass, test_data, layers, division, rho, eps, max_epoch)
%  INPUT:
%  train_data - MxN matrix, where N - number of samples, M - length of each sample (also size of input layer)
%  train_class - KxN array of classes, where K - number of classes for each sample
%  test_data - MxN_2 matrix for testing, where N_2 - number of test samples
%  layers - 1xC array of proposed architecture, consist of number of neurons for each layer. First number must be equal M, last one = K
%  division = 2x1 array, consist of propotions (in %) for division
%  train_data for train and validation datasets
%  rho - initial learning rate
%  eps - minimal MSE for finishing learning process
%  max_epoch - maximal number of epoches for finishing learning process
%  
%  OUTPUT:
%  test_class - KxN_2 array of classes for test_data
%  weights - optimal trained weights
%  full_error - MSE for validation set, which corresponds to the optimal weights
%  current_epoch - number of finished epoch
 
[M, N] = size(train_data);
 [testM, testN] = size(test_data);
 
 %normalization of data
 xx = train_data;
 for i=1:M
     xmax = max(xx(i,:));
     xx(i,:) = xx(i,:)/xmax;
 end
 
 tx = test_data;
 for i=1:testM
     txmax=max(tx(i,:));
     tx(i,:) = tx(i,:)/txmax;
 end
 tx(end+1,:) = 1;
 
 %divide dataset to "train" and "validation" set
 p = randperm(N);
 trainN = fix(N*division(1)/100);
 validationN = fix(N*division(2)/100);
 to_train = p(1:trainN);
 to_valid = p(1+trainN:1+trainN+validationN);
 x = xx(1:M, to_train); % "train"
 [xM,xN] = size(x);
 x(end+1,:) = 1;
 vx = xx(1:M, to_valid); % "validation"
 [vM,vN] = size(vx);
 vx(end+1,:) = 1;
 train_class = trainclass(:, to_train);
 validation_class = trainclass(:, to_valid);
 v_out = zeros(size(validation_class,1),vN);
 
 %weights initialization
 N_of_layers = length(layers);
 for i=1:N_of_layers-1
     weights{i}=randn(layers(i) + 1, layers(i+1));
 end
 optimal_weights = {};
 
 [classM, classN] = size(train_class);
 test_class = zeros(classM, testN);
 
 min_validation_mse = classM;
 validation_mse = 0;
 current_epoch = 0;
 flag = true;
 
 while flag
     full_error = 0;
     %feed forward
     for j = 1:xN
         computed_layers = {};
         computed_layers{1} = x(:, j);
         for i_layer=2:N_of_layers
             computed_layers{i_layer} = weights{i_layer-1}' * computed_layers{i_layer-1};
             for k=1:layers(i_layer)
                 computed_layers{i_layer}(k) = act(computed_layers{i_layer}(k));
             end
             if i_layer < N_of_layers
                 computed_layers{i_layer}( layers(i_layer)+1 ) = 1;
             end
         end
         
         out = computed_layers{end};
         
         current_error = out - train_class(:, j);
         full_error = full_error + sum(current_error.^2);
         
         deltas = {};
         p = N_of_layers-1;
         activation_vector = out;
         for k=1:length(activation_vector)
             activation_vector(k)=d_act(activation_vector(k));
         end
         deltas{p} = current_error .* activation_vector;
         for i=1:p-1
             activation_vector = computed_layers{p-i+1};
             for k=1:length(activation_vector)
                 activation_vector(k)=d_act(activation_vector(k));
             end
             deltas{p-i} = (weights{p-i+1} * deltas{p-i+1}) .* activation_vector;
             deltas{p-i} = deltas{p-i}(1:end-1);
         end
         
         % backpropagation
         for i=1:length(weights)
             weights{i} = weights{i} - rho * computed_layers{i} * deltas{i}';
         end
     end
     
     v_layers = {};
     validation_mse = 0;
     for j=1:vN
         v_layers{1} = vx(:, j);
         for i_layer=2:N_of_layers
             v_layers{i_layer} = weights{i_layer-1}' * v_layers{i_layer-1};
             for k=1:layers(i_layer)
                 % Activation function
                 v_layers{i_layer}(k) = act(v_layers{i_layer}(k));
             end
             if i_layer < length(layers)
                 v_layers{i_layer}( layers(i_layer)+1 ) = 1;
             end
         end
         v_out(:, j) = v_layers{end};
         validation_error = v_out(:, j) - validation_class(:, j);
         validation_mse = validation_mse + sum(validation_error.^2);
     end
     validation_mse = validation_mse / vN;
     % If found new minimum MSE on validation set, update best weights
     if (validation_mse < min_validation_mse)
         min_validation_mse = validation_mse;
         optimal_weights = weights;
     end
     
     current_epoch = current_epoch + 1;
     full_error = full_error/xN;
     
     if (current_epoch >= max_epoch) || (full_error < eps)
         flag = false;
     end
     
     rho = full_error/(10*classM);
     
     v_cls = zeros(size(v_out));
     v_cls(find(v_out>0.5))=1;
     accuracy = sum(v_cls==validation_class)/length(validation_class);
     fprintf('Current epoch: %04d \nMSE: %10.6f\nValidation MSE: %10.6f\nValidation accuracy: %10.6f\n\n', ...
         current_epoch, full_error, validation_mse, accuracy);
 end
 weights = optimal_weights;
 full_error = min_validation_mse;
     
 t_layers = {};
 for j=1:testN
     t_layers{1} = tx(:, j);
     for i_layer=2:N_of_layers
         t_layers{i_layer} = weights{i_layer-1}' * t_layers{i_layer-1};
         for k=1:layers(i_layer)
             t_layers{i_layer}(k) = act(t_layers{i_layer}(k));
         end
         if i_layer < N_of_layers
             t_layers{i_layer}( layers(i_layer)+1 ) = 1;
         end
     end
     test_class(:, j) = t_layers{N_of_layers};
 end
 
 
end

function f = act(x)
 %f = (1 + exp(x)).^(-1);
 %f = (sqrt(x.^2+1)-1)./2 - x;
 f = tanh(x);
end

function f = d_act(x)
 %f = act(x).*(1-act(x));
 %f = x./2*sqrt(x.^2+1) - 1;
 f = 1 - act(x).^2;
end

