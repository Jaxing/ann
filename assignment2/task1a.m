n = 2;
m = 100;
inputs = rand(n, 1000);

weights = -1 * (1+1) * rand(m, n);

for i=1:length(inputs)
    input = inputs(:, i);
    if input(1) > 0.5 && input(2) < 0.5
        r = rand/2;
        inputs(:, i) = [input(1) - 0.5, input(2) + 0.5];
    end
end

subplot(3,1,1)
plot(inputs(1,:), inputs(2,:))
indicies = 1:length(weights);

%----- Ordering Phase -----
iterations = 10^3;
sigma_null = 100;
learning_null = 0.1;
t_sigma = 300;


for i = 1:iterations
    sigma = sigma_null * exp(-(i-1)/t_sigma);
    learning_rate = learning_null * exp(-(i-1)/t_sigma);
    pattern = inputs(:, i);
    winning = winning_index(weights, pattern, indicies);
    
    delta = learning_rule(indicies, winning, pattern, weights, learning_rate, sigma);
    weights = weights + delta;
end
subplot(3,1,2);
plot(weights(:,1), weights(:,2));
%----- Conv Phase ------
iterations = 2 * 10^4;
sigma = 0.9;
leraning_rate = 0.01;

for i = 1:iterations
    it = mod(i, length(inputs));
    
    if it == 0
        it = length(inputs);
    end
    
    pattern = inputs(:, it);
    winning = winning_index(weights, pattern, indicies);
    
    delta = learning_rule(indicies, winning, pattern, weights, learning_rate, sigma);
    weights = weights + delta;
end
subplot(3,1,3);
plot(weights(:,1), weights(:,2));

function delta = learning_rule(indicies, winning_index, pattern, weights, learning_rate, neighborhood_sigma)
    tmp = @(x) neighborhood(x, winning_index, neighborhood_sigma);
    neighbors = arrayfun(tmp, indicies);
    
    delta = learning_rate * neighbors.' .* (pattern.' - weights);
end

function lambda = neighborhood(index, index_comp, sigma)
    lambda = exp(-abs(index - index_comp)^2 / (2 * sigma^2));
end

function index = winning_index(weights, pattern, indicies)
    dist = inf;
    for j =indicies
        weight = weights(j, :).';
        dist_tmp = norm(pattern - weight);
        if dist_tmp < dist
            dist = dist_tmp;
            winning= j;
        end
    end
    index = winning;
end