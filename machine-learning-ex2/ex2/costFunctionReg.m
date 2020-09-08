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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
J = - y' * log (h) - (1-y)' * log(1-h);
J = J / m; 

sz = size(theta,1);
temp = theta;
tempsum = 0;
for j = 2:sz
  temp(j) = temp(j) * temp(j);
  tempsum = tempsum+temp(j);
endfor
J = J + (tempsum * lambda) / (2*m);


grad = X' * (h-y);
grad = grad / m;

for j = 2:sz
  grad(j) = grad(j) + lambda * theta(j) / m;
endfor



% =============================================================

end
