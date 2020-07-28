function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta ;
y2 = h-y;
y2 = y2 .^2;
J1 = sum((1/(2*m)) * y2);

grad2 = X' * (h-y);
grad2 = grad2 / m;
theta (1) = 0;
grad1 = theta*(lambda/m);
grad = grad2 + grad1;

theta = theta.^2;
J2= sum(theta * lambda / (2*m));

J = J1+J2;



% =========================================================================

grad = grad(:);

end
