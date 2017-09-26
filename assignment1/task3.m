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

train_error = zeros(1,10)
val_error = zeros(1,10)
for lolipop=1:10
lolipop
learning_rate = 0.02;
beta = 1/2;

weights = rand(2,1)* 0.4 -0.2;
biase = rand * 2 - 1;

energy_train = zeros(1,10^3);
energy_validation = zeros(1,10^3);
iteration_range = 1:10^6;

l = 1;
for iteration = iteration_range
    
    i = mod(iteration, length(training_target));
    if i == 0
        i = length(training_target);
    end
    prediction = predict(training_input(i,:), weights, biase);
    if mod(iteration, 1000) == 0
        energy_train(l) = energy_function(training_target, training_input, weights, biase);
        energy_validation(l) = energy_function(validation_target, validation_input, weights, biase);
        l = l + 1;
    end
    weights = weights + update_all_weights(training_target(i), prediction, weights, training_input(i,:), biase);
    biase = biase + update_biase(training_target(i), prediction, weights, training_input(i,:), biase);
end
train_error(lolipop) = class_error(training_target, training_input, weights, biase)
val_error(lolipop) = class_error(validation_target, validation_input, weights, biase)


hold on
plot(1:10^3, energy_train);
plot(1:10^3, energy_validation);
xlabel('iterations')
ylabel('energy')
end
avg_err_train = mean(train_error)
min_err_train = min(train_error)
var_err_train = var(train_error)

avg_err_val = mean(val_error)
min_err_val = min(val_error)
var_err_val = var(val_error)
function c = class_error(targets, inputs, weights, biase)
    p = length(targets);
    summa = 0;
    
    for i=1:p
        prediction = predict(inputs(i, :), weights, biase);
        summa = summa + abs(targets(i) - sign(prediction));
    end
    c = summa/(2*p);
end

function f = predict(input, weights, biase)
    beta = 1/2;
    
    f = tanh(beta*(input * weights - biase));
end

function H = energy_function(target, input, weights, biase)
    summa = 0;
    p=length(target);
    for i=1:p
        prediction = predict(input(i,:), weights, biase);
        summa = summa + (target(i) - prediction)^2;
    end
    H = summa/2;
end

function b = update_biase(target, predicted, weights, input, biase)
    learning_rate = 0.02;
    beta = 1/2;
    
    b = learning_rate*((target - predicted))*(-sech(beta*input*weights - biase)^2 * beta);
end

function W = update_all_weights(target, predicted, weights, input, biase)
    learning_rate = 0.02;
    beta = 1/2;
    
    g_prime = sech(beta * input * weights - biase).^2 * beta;
    delta_error = (target - predicted) * g_prime;
    
    W = learning_rate * delta_error .* input.';
end