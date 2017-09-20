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

weights_hid = rand(hidden_units, input_units)* 0.4 - 0.2;
biase_hid = rand(hidden_units,1);

weights_out = rand(output_units, hidden_units)* 0.4 - 0.2;
biase_out = rand*2 - 1;

energy_train = zeros(1,10^3);
energy_validation = zeros(1,10^3);
iteration_range = 1:10^3;

train_error = class_error(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid)
val_error = class_error(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid)
tic

for iteration = iteration_range
    tic
    i = mod(iteration, length(training_target));
    if i == 0
        i = length(training_target);
    end
    
    predict_hid = predict(training_input(i,:).', weights_hid, biase_hid);
    predict_out = predict(predict_hid, weights_out, biase_out);
    
    l = energy_function(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid);
    energy_train(i) = l;
    energy_validation(i) = energy_function(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid);
    
    weights_out = update_weights_out(training_target(i,:), predict_out, weights_out, predict_hid, biase_out)
    weights_hid = update_weights_hid(training_target(i,:), predict_out, predict_hid, weights_out, weights_hid, training_input(i,:), biase_out, biase_hid)
    one_it = toc;
end
toc
train_error = class_error(training_target, training_input, weights_out, weights_hid, biase_out, biase_hid)
val_error = class_error(validation_target, validation_input, weights_out, weights_hid, biase_out, biase_hid)

hold on
plot(iteration_range, energy_train);
plot(iteration_range, energy_validation)

function H = energy_function(target, input, weights, weights_hid, biase, biase_hid)
    summa = 0;
    p=length(target);
    for i=1:p
        predict_hid = predict(input(i,:).', weights_hid, biase_hid);
        predict_out = predict(predict_hid, weights, biase);
        summa = summa + (target(i) - predict_out).^2;
    end

    H = summa/2;
end

function c = class_error(targets, inputs, weights, weights_hid, biase, biase_hid)
    p = length(targets);
    summa = 0;
    
    for i=1:p
        
        predict_hid = predict(inputs(i,:).', weights_hid, biase_hid);
        predict_out = predict(predict_hid, weights, biase);
        
        summa = summa + abs(targets(i) - sign(predict_out));
    end
    c = summa/(2*p);
end

function f = predict(input, weights, biase)
    tmp = (weights * input);
    f = activation(tmp - biase);
end

function W = update_weights_hid(targets, predicted, predicted_hid, weights_out, weights_hid, input, biase_out, biase_hid)
    learning_rate = 0.02;
    beta = 1/2;
    
    delta_error_out = sech(beta * (weights_out * predicted_hid - biase_out))^2* beta * predicted_hid * (targets - predicted);
    delta_error = input.' * sech(beta * (input * weights_hid.' - biase_hid.'))^2 * beta * (weights_out.' .* delta_error_out);
    
    W = (learning_rate * weights_hid.' .* delta_error).';
end

function W = update_weights_out(targets, predicted, weights_out, input, biase)
    learning_rate = 0.02;
    beta = 1/2;
    
    delta_error = sech(beta * (weights_out * input - biase))^2 * beta * input * (targets - predicted);
    
    W = (learning_rate * weights_out.' .* delta_error).';
end

function g = activation(b)
    beta = 1/2;
    g = tanh(beta*b);
end