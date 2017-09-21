training_data = importdata('train_data.m');
validation_data = importdata('validation_data.m');

training_input = training_data(:,1:2);
training_target = training_data(:,3);

validation_input = validation_data(:,1:2);
validation_target = validation_data(:,3);

%-------- Normalising training data

average = mean(training_input);
variance = var(training_input);

training_input(:,1) = (training_input(:,1)-average(1))/sqrt(variance(1));
training_input(:,2) = (training_input(:,2)-average(2))/sqrt(variance(2));

%-------- Normalising validation data

average = mean(validation_input);
variance = var(validation_input);

validation_input(:,1) = (validation_input(:,1)-average(1))/sqrt(variance(1));
validation_input(:,2) = (validation_input(:,2)-average(2))/sqrt(variance(2));

%------- Implemention below
input_units = 2;
hidden_units = 4;
output_units = 1;

training_class_error = zeros(1,10);
validation_class_error = zeros(1,10);

for lolipop=1:10
lolipop
weights_hid = rand(hidden_units, input_units)* 0.4 - 0.2;
biase_hid = rand(hidden_units,1);

weights_out = rand(output_units, hidden_units)* 0.4 - 0.2;
biase_out = rand*2 - 1;

energy_train = zeros(1,10^3);
energy_validation = zeros(1,10^3);
iteration_range = 1:10^6;

l=1;
for iteration = iteration_range
    i = mod(iteration, length(training_target));
    if i == 0
        i = length(training_target);
    end
    
    predict_hid = predict(training_input(i,:), weights_hid, biase_hid);
    predict_out = predict(predict_hid, weights_out, biase_out);
    
    if mod(iteration, 1000) == 0
        energy_train(l) = energy_function(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid);
        energy_validation(l) = energy_function(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid);
        l = l + 1;
    end
    
    
    delta_out = update_weights_out(training_target(i,:), predict_out, weights_out, predict_hid, biase_out);
    delta_out_biase = update_biase_out(training_target(i,:), predict_out, weights_out, predict_hid, biase_out);    
    
    
    delta_hid = update_weights_hid(training_target(i,:), predict_out, predict_hid, weights_out, weights_hid, training_input(i,:), biase_out, biase_hid);
    delta_hid_biase = update_biase_hid(training_target(i,:), predict_out, predict_hid, weights_out, weights_hid, training_input(i,:), biase_out, biase_hid);
    
    weights_out = weights_out + delta_out;
    biase_out = biase_out + delta_out_biase;
    
    weights_hid = weights_hid + delta_hid;
    biase_hid = biase_hid + delta_hid_biase;
end
toc
training_class_error(lolipop) = class_error(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid);
validation_class_error(lolipop) = class_error(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid);

toc
hold on
plot(1:10^3, energy_train);
plot(1:10^3, energy_validation);
xlabel('1000 iterations')
ylabel('energy')
end
avg_err_train = mean(training_class_error)
min_err_train = min(training_class_error)
var_err_train = var(training_class_error)

avg_err_val = mean(validation_class_error)
min_err_val = min(validation_class_error)
var_err_val = var(validation_class_error)
function H = energy_function(target, input, weights, weights_hid, biase, biase_hid)
    summa = 0;
    p=length(target);
    for i=1:p
        predict_hid = predict(input(i,:), weights_hid, biase_hid);
        predict_out = predict(predict_hid, weights, biase);
        summa = summa + (target(i) - predict_out).^2;
    end

    H = summa/2;
end

function c = class_error(targets, inputs, weights, weights_hid, biase, biase_hid)
    p = length(targets);
    summa = 0;
    
    for i=1:p
        
        predict_hid = predict(inputs(i,:), weights_hid, biase_hid);
        predict_out = predict(predict_hid, weights, biase);
        
        summa = summa + abs(targets(i) - sign(predict_out));
    end
    c = summa/(2*p);
end

function f = predict(input, weights, biase)
    f = activation(weights * input.' - biase).';
end

function W = update_weights_hid(targets, predicted, predicted_hid, weights_out, weights_hid, input, biase_out, biase_hid)
    learning_rate = 0.02;
    beta = 1/2;
    
    g_prime_out = derivative_activation(weights_out * predicted_hid.' - biase_out); %sech(beta * (weights_out * predicted_hid.' - biase_out)).^2 * beta * predicted_hid;
    g_prime_hid = derivative_activation(weights_hid * input.' - biase_hid); %sech(beta * (weights_hid * input.' - biase_hid)).^2 * beta;
    
    delta_error_out = (targets - predicted) * g_prime_out;
    delta_error = (delta_error_out .* weights_out).' .* g_prime_hid;
    
    W = learning_rate * delta_error * input;
end

function b = update_biase_hid(targets, predicted, predicted_hid, weights_out, weights_hid, input, biase_out, biase_hid)
    learning_rate = 0.02;
    beta = 1/2;
    
    g_prime_out = derivative_activation(weights_out * predicted_hid.' - biase_out); %-sech(beta * (weights_out * predicted_hid.' - biase_out)).^2 * beta;
    g_prime_hid = derivative_activation(weights_hid * input.' - biase_hid);
    
    delta_error_out = (targets - predicted) * g_prime_out;
    delta_error_hid = delta_error_out .* weights_out.' .* -g_prime_hid;
    
    b = learning_rate * delta_error_hid;
end

function W = update_weights_out(targets, predicted, weights_out, input, biase)
    learning_rate = 0.02;
    
    g_prime = derivative_activation(weights_out * input.' - biase); %sech(beta * (weights_out * input.' - biase))^2 * beta;
    delta_error =(targets - predicted) * g_prime;
    
    W = learning_rate * delta_error .* input;
end

function b = update_biase_out(targets, predicted, weights_out, input, biase)
    learning_rate = 0.02;
    
    g_prime = derivative_activation(weights_out * input.' - biase);
    
    b = learning_rate * (targets - predicted) * -g_prime;
end

function g = derivative_activation(b)
    beta = 1/2;
    g = sech(beta*b).^2 *beta;
end

function g = activation(b)
    beta = 1/2;
    g = tanh(beta*b);
end