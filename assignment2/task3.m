data = importdata('data_task3.m');
target_values = data(:, 1);
input_comp = data(:, [2, 3]);

class_one = (input_comp(find(target_values == 1),:));
class_two = (input_comp(find(target_values == -1),:));

hold on
scatter(class_one(:,1),class_one(:,2),'y');
scatter(class_two(:,1),class_two(:,2),'g');


%------ Network implemntation begins here.
k = 10;

class_errors = zeros(1,20);
wc = zeros(k,2);
ws = zeros(1, k);
b = 0;
min_error = inf;
tic
for j=1:20

    indicies = 1:k;

    weights_comp = 2 * rand(k, 2) - 1;

    %------ Train competative network
    for iteration = 1:10^5
        i = mod(iteration, length(target_values));
        if i == 0
            i = length(target_values);
        end
        
        
        pattern = input_comp(randi(length(input_comp)),:);
        weights_comp;
        [winning_neuron, i_0] = max(activation_comp(pattern, weights_comp));
        delta_comp_weights = update_weights(pattern, weights_comp(i_0, :));
        weights_comp(i_0, :) = weights_comp(i_0, :) + delta_comp_weights;
    end
    plot(weights_comp(:,1),weights_comp(:,2), 'ko');
    drawnow
    %------ Compute input to supervised network
    input_sup = zeros(length(target_values), k);
    for i = 1:length(target_values)
        pattern = input_comp(i, :);
        res = activation_comp(pattern, weights_comp);
        input_sup(i, :) = res;
    end

    biase =  2 * rand - 1;
    %biase = 0;
    weights_sup = 2 * rand(1, k) - 1;
    input_sup;
    %------- Train supervised network
    for iteration = 1:3*10^3
        i = mod(iteration, length(target_values));
        if i == 0
            i = length(target_values);
        end
        rand_index = randi(length(input_sup));
        pattern = input_sup(rand_index,:);
        prediction = predict(pattern, weights_sup, biase);
        
        
        delta_weights = update_all_weights(target_values(i), prediction, weights_sup, pattern, biase);
        delta_biase = update_biase(target_values(i), prediction, weights_sup, pattern, biase);
        
        weights_sup = weights_sup + delta_weights;
        biase = biase + delta_biase;
    end
    plot(weights_sup(:, 1) - biase, weights_sup(:, 2) - biase, 'bo')
    drawnow
    ce = class_error(target_values, input_comp, weights_comp, weights_sup, biase)
    class_errors(j) = ce;
    if min_error > ce 
        min_error = ce;
        wc = weights_comp;
        ws = weights_sup;
        b  = biase;
    end
end
avg_class_error = mean(class_errors)

x_range = -15:0.01:25;
y_range = -10:0.01:15;

% plot the predicted classes for points outside of the dataset

for i=1:10000
    pattern = [datasample(x_range,1) datasample(y_range,1)];
    
    prediction_comp = activation_comp(pattern, wc);
    prediction_sup = predict(prediction_comp, ws, b);
    
    if sign(prediction_sup) > 0
        plot(pattern(1), pattern(2), 'yo');
    else
        plot(pattern(1), pattern(2), 'go');
    end
    drawnow
end

toc
function f = predict(input, weights, biase)
    beta = 1/2;
    f = tanh(beta*(input * weights.' - biase));
end

function b = update_biase(target, predicted, weights, input, biase)
    learning_rate = 0.1;
    beta = 1/2;
    g_prime = derivative_activation(input * weights.' - biase);
    b = learning_rate*(target - predicted)* -g_prime;
end

function W = update_all_weights(target, predicted, weights, input, biase)
    learning_rate = 0.1;
    beta = 1/2;
    
    g_prime = derivative_activation(input * weights.' - biase);
    delta_error = (target - predicted) * g_prime;
    
    W = learning_rate * delta_error.' .* input;
end

function g = derivative_activation(b)
    beta = 1/2;
    g = sech(beta*b).^2 *beta;
end

function f = gaussian(pattern, weights)
    res = (pattern - weights);
    res = res(:, 1).^2 + res(:, 2).^2;
    f = exp(-res/2).';
end

function g = activation_comp(pattern, weights)
    hej = gaussian(pattern, weights);
    summa = sum(hej);
    g = hej/summa;
end

function delta_w = update_weights(pattern, weights)
    learning_rate = 0.02;
    delta_w = learning_rate*(pattern - weights);
end

function c = class_error(targets, patterns, weights_comp, weights_sup, biase)
    p = length(targets);
    summa = 0;
   
    patterns;
    
    for i=1:p
        pattern = patterns(i, :);
        weights_comp;
        prediction_comp = activation_comp(pattern, weights_comp);
        weights_sup;
        biase;

        prediction_sup = predict(prediction_comp, weights_sup, biase);
        
        summa = summa + abs(targets(i) - sign(prediction_sup));
    end
    c = summa/(2*p);
end
