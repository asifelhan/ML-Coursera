function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = sigmoid (X * theta);
l1 = log (h); l2 = log (1-h);
J = (y' * l1) + ((1-(y'))* l2);
J = J* (-1);
J = (1/m) * J;
grad = h - y;
grad = X' * grad;
grad = (1/m) * grad; 



J1 = (theta.^2 )/(2*m);
J1 = J1 * lambda;
J = J + J1;

theta (1) = 0;

grad1 = (lambda/m) * theta;
grad = grad + grad1; 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
