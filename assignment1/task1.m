tic
hold on
plot_theo()
plot_num()
xlabel('alpha = p/N')
ylabel('pError')
toc

function plot_num()
    N = 200;
    ps = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400];

    n = length(ps);

    errors = zeros(n,1);
    pNs = errors;

    for i=1:n        
        x=ps(i)/N;
        pNs(i) = x;
        errors(i) = error_num(ps(i), N);
    end
    scatter(pNs, errors)
end

function f = error_num(p,N)    
    patterns = generate_n_random_patterns(N, p);
    weights = create_weight_matrix(patterns);
    patterns = generate_n_random_patterns(N, p);
    patterns_left = 10000;
    summa = 0;
    count = 0;
    while patterns_left > 0
        weights = create_weight_matrix(patterns);
        summa_prime = 0;
        for i=1:length(patterns(1, :))
           s = update(patterns(:, i), weights);
           summa_prime = summa_prime + number_of_bits_differ(s, patterns(:, i));
        end
        summa = summa + summa_prime/(N * length(patterns(1, :)));
        count = count + 1;
        patterns_left = patterns_left - length(patterns(1, :));
        if patterns_left < p
            patterns = generate_n_random_patterns(N, patterns_left);
        end
    end
    f = summa/count;
end

function p = generate_n_random_patterns(neurons, n)
    patterns = zeros(neurons, n);
    for j=1:n
        pattern = generate_random_pattern(neurons);
        patterns(:, j) = pattern;
    end
    
    p = patterns;
end

function h = number_of_bits_differ(p_mu, p)
    d = 0;
    for i=1:length(p)
        if p(i)*p_mu(i) < 0
            d = d + 1;
        end
    end
    h = d;
end

function plot_theo()
    N = 200;
    ps = 1:1:400;

    n = length(ps);

    errors = zeros(n,1);
    pNs = errors;

    for i=1:n
        x=ps(i)/N;
        pNs(i) = x;
        errors(i) = pError(ps(i), N);
    end
    plot(pNs, errors)
end

function f = pError(p, N)
    fun = @(x) exp(-x.^2);
    erf = @(z) (2/sqrt(pi)) * integral(fun, 0, z);
    f = (1/2) * (1 - erf(sqrt(N/(2 * p))*(1 + (p/N))));
end

function s = update(pattern, weights)
    s = sign(weights * pattern);
end

function f = generate_random_pattern(neurons)
    pattern = randi([0 1], neurons, 1);
    pattern(pattern==0) = -1;
    f = pattern;
end

function w = create_weight_matrix(patterns)
    N = length(patterns(:, 1));
    p = length(patterns(1, :));
    
    matrix = patterns * patterns.';
    w = 1/N * matrix;
end