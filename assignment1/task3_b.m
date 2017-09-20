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


learning_rate = 0.02;
beta = 1/2;

weights_hid = rand(hidden_units, input_units)* 0.4 - 0.2;
biase_hid = rand(hidden_units,1);

weights_out = rand(output_units, hidden_units)* 0.4 - 0.2;
biase_out = rand*2 - 1;

energy_train = zeros(1,10^3);
energy_validation = zeros(1,10^3);
iteration_range = 1:10^6;

train_error = class_error(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid)
val_error = class_error(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid)
tic
l = 1;
for iteration = iteration_range
    i = mod(iteration, length(training_target));
    if i == 0
        i = length(training_target);
    end
    prediction = predict(training_input(i,:), weights_out, weights_hid, biase_out, biase_hid);
    prediction_hid = zeros(hidden_units, 1);
    
    for j=1:hidden_units
        prediction_hid(j) = predict_hidden_neuron(training_input(i, :), weights_hid(j,:), biase_hid(j));
    end
    
    if mod(iteration, 1000) == 0
        energy_train(l) = energy_function(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid);
        energy_validation(l) = energy_function(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid);
        l = l + 1;
    end
    
    for k=1:output_units
        for m=1:hidden_units
            weights_out(k,m) = weights_out(k,m) + update_weight(m,training_target(i), prediction, weights_out, biase_out(k), prediction_hid);
            biase_out(k) = biase_out(k) + update_biase(training_target(i), prediction, weights_out, biase_out(k), prediction_hid);
        end
    end
    
    for k=1:hidden_units
        for m=1:input_units
            lol = update_weight_hidden_layer(k,m,training_target(i), prediction, prediction_hid, weights_out, weights_hid, biase_out, biase_hid(k), training_input(i,:));
            weights_hid(k,m) = weights_hid(k,m) + lol;
            biase_hid(k) = biase_hid(k) + update_biase_hidden_layer(k, training_target(i), prediction, prediction_hid, weights_out, biase_out, biase_hid(k), training_input(i, :));
        end
    end
    
    one_it = toc;
end
toc
train_error = class_error(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid)
val_error = class_error(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid)


hold on
plot(1:10^3, energy_train);
plot(1:10^3, energy_validation);

function c = class_error(targets, inputs, weights, weights_hid, biase, biase_hid)
    p = length(targets);
    summa = 0;
    
    for i=1:p
        prediction = predict(inputs(i, :), weights, weights_hid, biase, biase_hid);
        summa = summa + abs(targets(i) - sign(prediction));
    end
    c = summa/(2*p);
end

function f = predict_hidden_neuron(input, weights, biase)
    beta = 1/2;
    s=0;
    
    for i=1:length(input)
        s = s + tanh(beta*(input(i)*weights(i)-biase));
    end
    
    f = s;
end

function f = predict(input, weights, weights_hid, biase, biase_hid)
    summa = 0;
    beta = 1/2;
    
    for i=1:length(weights_hid(:,1))
        prediction = predict_hidden_neuron(input, weights_hid(i, :), biase_hid(i));
        summa = summa + tanh(beta* (prediction * weights(i) -biase));
    end
    f = summa;
end

function H = energy_function(target, input, weights, weights_hid, biase, biase_hid)
    summa = 0;
    p=length(target);
    for i=1:p
        prediction = predict(input(i,:), weights, weights_hid, biase, biase_hid);
        summa = summa + (target(i) - prediction)^2;
    end

    H = summa/2;
end
function b = update_biase_hidden_layer(output_index,target, predicted, predicted_hid, weights_out, biase_out, biase, input)
    learning_rate = 0.02;
    beta = 1/2;
    
    g_prime_out = sech(beta * (weights_out * predicted_hid - biase_out))*beta;
    g_prime_hid = sech(beta * (weights_out * input - biase))*beta;
    
    delta_error_out = (target - predicted) * g_prime_out;
    delta_error_hid = delta_error_out * weights_out(output_index) * g_prime_hid;
    
    b = learning_rate * biase * delta_error_hid;
    
    %delta = (target - predicted) * weights_out(:, output_index) * sech(beta * (weights(output_index, :) * input.' - biase)) * beta;
    
    %b = learning_rate * delta;
end

function b = update_biase(target, predicted, weights, biase, input_from_hid)
    learning_rate = 0.02;
    beta = 1/2;
    
    g_prime = sech(beta * (weights * input_from_hid - biase)) * beta;
    
    delta = (target - predicted) * g_prime; %might need to be multiplied by input_from_hid
    
    b = learning_rate * delta ;
end

function W = update_weight_hidden_layer(output_index, input_index, target, predicted, predicted_hid, weights_out, weights, biase_out, biase_hid, input)
    learning_rate = 0.02;
    beta = 1/2;
    
    weights
    input
    g_prime_out = sech(beta * (weights_out * predicted_hid - biase_out)) * beta * predicted_hid(output_index);
    g_prime_hid = sech(beta * (weights * input - biase_hid)) * beta * input(input_index);
    
    
    delta_error_out = g_prime_out * (target - predicted);
    delta_error_hid = g_prime_hid * weights_out(ouput_index) * delta_error_out;
    
    W = learning_rate * weights(input_index) * delta_error_hid;
    %g_prime_out = sech(beta * (weights_out * input_from_hid - biase_out)) * beta * predicted_hid(output_index);%not sure about the last input factor
    %g_prime_hid = sech(beta * (weights * input - biase_hid)) * beta * input(input_index);%not sure about the last input factor
    
    %delta = (target - predicted) * g_prime_out * weights_out(:, output_index) * g_prime_hid;
    
    %W = learning_rate * delta * input(input_index);
end

function W = update_weight(input_index, target, predicted, weights, biase, input_from_hid)
    learning_rate = 0.02;
    beta = 1/2;
    
    g_prime = sech(beta * (weights * input_from_hid - biase)) * beta * input_from_hid(input_index);%not sure about the last input factor
    delta = (target - predicted) * g_prime;
    W = learning_rate * weights(input_index) * delta 
end