tic
hold on
scatter(0,0)
for i=1:20
    main()
end
toc

function main()
    N = 200;
    p = 40;
    beta = 50;
    patterns = generate_n_random_patterns(N, p);
    time_steps = 0:1000;
    weights_prim = weights(patterns);
    pattern = patterns(:, 1);
    prev_state = pattern;
    result = zeros(1, length(time_steps));
    i = 1;
    for t=time_steps
        
        res = order_parameter(pattern, prev_state, beta, weights_prim);
        result(i) = res{1,1};
        prev_state = res{1,2};
        
        i = i + 1;
    end
    plot(time_steps, result)
end

function m = order_parameter(fed_pattern, prev_state, beta, weights)
    N = length(fed_pattern);
    summa = 0;
    state = zeros(length(fed_pattern), 1);
    
    for i=1:N
        state(i) = update(beta, i, prev_state, weights);
        summa = summa + state(i) * fed_pattern(i);
    end
    m = {summa/N, state};
end

function s = update(beta, index, prev_state, weights_prime)
    
    b_i = coolfunction(weights_prime(:, index),prev_state);
    probabillity = g(b_i, beta);
    random = rand;
    
    if random <= probabillity
       s = 1; 
    else
       s = -1;
    end
end

function w = weights(patterns)
    N = length(patterns(:, 1));
    
    matrix = patterns * patterns.';
    for i=1:N
        matrix(i,i) = 0;
    end
    w = 1/N * matrix;
end

function f = coolfunction(weights_prime, prev_state)
    f = weights_prime.' * prev_state;
end

function f = g(b_i, beta)
    f = 1 /(1 + exp(-2*beta*b_i));
end

function f = generate_random_pattern(neurons)
    pattern = randi([0 1], neurons, 1);
    pattern(pattern==0) = -1;
    f = pattern;
end

function p = generate_n_random_patterns(neurons, n)
    patterns = zeros(neurons, n);
    for j=1:n
        pattern = generate_random_pattern(neurons);
        patterns(:, j) = pattern;
    end
    
    p = patterns;
end