function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    t1 = theta(1);
    t2 = theta(2);

    X2 = X(:,2);

    s1 = 0;
    for iter2 = 1:m
      s1 = s1 + (t1 + t2*X2(iter2) - y(iter2));
    end
    

    s2 = 0;
    for iter2 = 1:m
      s2 = s2 + (t1 + t2*X2(iter2) - y(iter2)) * X2(iter2);
    end


    t1 = t1 - alpha * (1/m) * s1;
    t2 = t2 - alpha * (1/m) * s2;
    theta = [t1;t2;];








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
