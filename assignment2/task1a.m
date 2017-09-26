n = 2;
m = 100;
input = rand(n, 1000);

weights = rand(m, n);

for i=1:length(weights)
    weight = weights(i, :);
    if weight(1) > 0.5 && weight(2) < 0.5
        r = rand/2;
        weights(i, :) = [weight(1) * r, weight(2) * (2-r)];
    end
end

subplot(2,1,1)
plot(weights(:,1), weights(:,2))
indicies = 1:length(weights);

%----- Ordering Phase -----
iterations = 10^3;
sigma_null = 100;
learning_null = 0.1;
t_sigma = 300;


for i = 1:iterations
    sigma = sigma_null * exp(-(i-1)/t_sigma);
    learning_rate = learning_null * exp(-(i-1)/t_sigma);
    dist = inf;
    winning_index = 0;
    pattern = input(:, i);
    
    for j =indicies
        weight = weights(j, :).';
        dist_tmp = norm(pattern - weight);
        if dist_tmp < dist
            dist = dist_tmp;
            winning_index = j;
        end
    end
    delta = learning_rule(indicies, winning_index, pattern, weights, learning_rate, sigma);
    weights = weights + delta;
end
subplot(2,1,2);
weights(:,1);
weights(:,2);
plot(weights(:,1), weights(:,2));
%----- Conv Phase ------


function delta = learning_rule(indicies, winning_index, pattern, weights, learning_rate, neighborhood_sigma)
    tmp = @(x) neighborhood(x, winning_index, neighborhood_sigma);
    neighbors = arrayfun(tmp, indicies);
    
    delta = learning_rate * neighbors.' .* (pattern.' - weights);
end

function lambda = neighborhood(index, index_comp, sigma)
    lambda = exp(-abs(index - index_comp)^2 / (2 * sigma^2));
end