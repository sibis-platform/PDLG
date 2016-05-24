% Evaluating the logistic regression function

function [f,expf] = fun(A,x)
n = size(A,1); expf = zeros(n,1);
tmp = -A*x;
I = find(tmp>50); J = find(tmp<=50);
expf(J) = exp(tmp(J));
expf(I) = 1e20;
f = (sum(log(1+expf(J))) + sum(tmp(I)))/n;
