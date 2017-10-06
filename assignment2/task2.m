m = 1;
n = 2;

%---- uncentered mean

inputs = importdata('data_task2.m');
weights = -1 * (1+1) * rand(m, n);

iterations = 2 * 10^4;

weights_over_time = zeros(iterations+1, 1);
weights_over_time(1, :) = norm(weights);
for i = 1:iterations
    it = mod(i, length(inputs));
    
    if it == 0
        it = length(inputs);
    end
    
    prediction = predict(inputs(it,:), weights);
    weights = weights + learning_rule(inputs(it,:), prediction, weights);
    weights_over_time(i+1) = norm(weights);
end

subplot(2,2,1);
hold on
scatter(0,0)
title('Non-centered mean')
plot(0:iterations, weights_over_time);
xlabel('Iterations')
ylabel('Norm of weights')

subplot(2,2,3);
plot(inputs(:,1), inputs(:,2));

%----- Centered mean

avg = mean(inputs);

inputs(:, 1) = inputs(:, 1) - avg(1);
inputs(:, 2) = inputs(:, 2) - avg(2);

weights = -1 * (1+1) * rand(m, n);

weights_over_time = zeros(iterations+1, 1);
weights_over_time(1, :) = norm(weights);
for i = 1:iterations
    it = mod(i, length(inputs));
    
    if it == 0
        it = length(inputs);
    end
    
    prediction = predict(inputs(it,:), weights);
    weights = weights + learning_rule(inputs(it,:), prediction, weights);
    weights_over_time(i+1, :) = norm(weights);
end

subplot(2,2,2);
hold on
title('Centered mean')
scatter(0,0)
plot(0:iterations, weights_over_time);
xlabel('Iterations')
ylabel('Norm of weights')

subplot(2,2,4);
plot(inputs(:,1), inputs(:,2));

function f = predict(pattern, weights)
    f = weights * pattern.';
end

function delta = learning_rule(pattern, predicted, weights)
    learning_rate = 0.001;
    delta = learning_rate * predicted *(pattern - predicted * weights);
end