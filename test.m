% script for testing the penalty decomposition (PD) method for solving 
% the group cardinality constrained logistic regression  

clear all

%% Generate data
rand('seed',100);
randn('seed',100);
n = 1000 ; p = 100; 
Xtype = 'SM';
eps =1e-4;
maxit = 10000;
I = randperm(n); b = ones(n,1);
b(I(1:n/2)) = -1; X = zeros(n,p);
for (j = 1:n)
    X(j,:) = b(j)*rand + randn(1,p);
end

% Generating groups
for i = 1:50
    group{i} = [i i+50];
end
k = 0.1*p;

% Generating initial point
I = randperm(50);
x0 = zeros(p,1);
for i = 1:k
    x0(group{I(i)}) = rand(length(group{I(i)}),1);
end
x0 = [x0;1];

%% Solve the problem by PD method
x = PD_logreg_group(X,b,k,group,'tol',eps,'maxit',maxit,'init',x0);

