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

learning_rate = 0.02;
beta = 1/2;

weights = rand(2,1)* 0.4 -0.2;
biase = rand * 2 - 1;

energy_train = zeros(1,10^6);
energy_validation = zeros(1,10^6);
iteration_range = 1:10^6;

train_error = class_error(training_target, training_input, weights, biase)
val_error = class_error(validation_target, validation_input, weights, biase)

for iteration = iteration_range
    tic
    i = mod(iteration, length(training_target));
    if i == 0
        i = length(training_target);
    end
    prediction = predict(training_input(i,:), weights, biase);
    energy_train(iteration) = energy_function(training_target, training_input, weights, biase);
    energy_validation(iteration) = energy_function(validation_target, validation_input, weights, biase);
    weights(1,1) = weights(1,1) + update_weights(1,1,training_target(i), prediction, weights, training_input(i,:), biase);
    weights(2,1) = weights(2,1) + update_weights(1,2,training_target(i), prediction, weights, training_input(i,:), biase);
    biase = biase + update_biase(1,1,training_target(i), prediction, weights, training_input(i,:), biase);
    one_it = toc;
end
train_error = class_error(training_target, training_input, weights, biase)
val_error = class_error(validation_target, validation_input, weights, biase)


hold on
plot(iteration_range, energy_train);
plot(iteration_range, energy_validation);

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
    s=0;
    
    for i=1:length(input)
        s = s + tanh(beta*(input(i)*weights(i)-biase));
    end
    
    f = s;
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

function b = update_biase(output_index, input_index, target, predicted, weights, input, biase)
    learning_rate = 0.02;
    beta = 1/2;
    
    b = learning_rate*((target - predicted))*sech(beta*input*weights - biase)^2 * beta;
end

function W = update_weights(output_index, input_index, target, predicted, weights, input, biase)
    learning_rate = 0.02;
    beta = 1/2;
    
    W = learning_rate*((target - predicted))*sech(beta*input*weights - biase)^2 * beta * input(:, input_index) * input(:, input_index);

    %W =(target(output_index)-predicted(output_index))*weights(input_index, output_index)*input(:,input_index);
end