hold on
main()

function main()
    N = 200;
    p = 5;
    beta = 2;
    patterns = generate_n_random_patterns(N, p);
    time_steps = 0:10:100;
    pattern = patterns(:, 1);
    weights = get_weights(patterns);
    prev_state = pattern;
    result = zeros(1, length(time_steps));
    i = 1;
    for t=time_steps
        if t == 0
            result(i) = (pattern.' * pattern)/N;
            i = i + 1;
            continue
        end
        res = traveling_mean(N, beta, t-10, t, prev_state, pattern, weights);
        result(i) = res{1,1};
        prev_state = res{1,2};
    end
    plot(time_steps, result)
end

function m_mu = traveling_mean(N, beta, start_time, end_time, start_state, pattern, weights)
    summa = 0;
    prev_state = start_state;
    
    for t=start_time:end_time
        if t ~= start_time
            prev_state = update_whole_state(beta, prev_state, weights);
        end
        op = orderparameter(N, prev_state, pattern);
        summa = summa + op;
    end
    m_mu = {summa/(end_time - start_time), prev_state};
end

function m = orderparameter(N, state, pattern)    
    m = (state.' * pattern)/N;
end

function s = update_whole_state(beta, prev_state, weights)
    for i=1:length(prev_state)
        prev_state(i) = update_state(i, beta, prev_state, weights);
    end
    s = prev_state;
end

function s = update_state(index, beta, prev_state, weights)
    g = 1/(1+exp(-2*beta*b_i(index, prev_state, weights)));
    random = rand;
    
    if g >= random
        s = 1;
        return
    end
    s=-1;
end

function b = b_i(index, prev_state, weights)
    summa = 0;
    for j=1:length(prev_state)
        summa = summa + weights(index,j) * prev_state(j);
    end
    b = summa;
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

function w = get_weights(patterns)
    N = length(patterns(:, 1));
    
    matrix = patterns * patterns.';
    for i=1:N
        matrix(i,i) = 0;
    end
    w = 1/N * matrix;
end