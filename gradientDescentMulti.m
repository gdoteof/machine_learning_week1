function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X,2);
new_theta = zeros(n,1);
sums = zeros(n,1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
     
    %update all thetas simultaneously
    %so, for each feature
    for iter_n = 1:n % feature

      _sum = 0;
      for iter_m = 1:m % 
        _sample = X(iter_m,:);             % the mth row of 
        _htheta = (_sample * theta);      % h(_sample)   our hypothesis for the sample based on theta
        _answer = y(iter_m);               % y(_sample)   the real answer
        _diff = _htheta - _answer;         % h(_sample) - y(_sample)   the difference between our hypothesis and real anaswer
        _feature = _sample(iter_n);        % nth feature of _sample    the feature this theta deals with
        _sum += _diff * _feature;
      end
      new_theta(iter_n) = theta(iter_n) - alpha* (1/m) * _sum;
    end

    %set new theta
    theta = new_theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end









end

