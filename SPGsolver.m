%
% See LICENSE file distributed along with the package for the copyright and license terms.
%
% solve the following problem via SPG 
%  min e'*f(Ax)/n + |x-c|^2*(rho/2)
% where e is the all-ones vector, f(x) = log(1+exp(-x)),

function [x,d] = SPGsolver(A,c,rho,x,d)
% Initialization
eps = 1e-6;
gamma = 1e-4;
M = 2;
alphamin = 1e-8;
alphamax = 1;
f_1 = -inf*ones(M,1);
[f,expf] = fun(A,x);
f = f + 0.5*rho*norm(x-c)^2;
g = grad(A,c,rho,x,expf);

err = norm(g);

alphas = 1;

f_1(1) = f;
k = 1;iter_in = 1;

% Main Algorithm
while err >= eps
    f_max = max(f_1);
    d = -alphas*g;
    delta = sum(d.*g);
    [x_new,f_new,g_new] = linesearch(A,c,rho,x,d,delta,f_max,gamma);  
    f_1(mod(k,M)+1) = f_new;
    dx = x_new - x;
    dg = g_new - g;
    xdotg = sum(dx.*dg);
    xsqr =  norm(dx)^2;
    alphas = max(alphamin,min(alphamax,xsqr/xdotg));
    x = x_new;
    g = g_new;
    err = abs(f - f_new)/max(abs(f),1);
    f = f_new;
    iter_in = iter_in + 1; 
    k = k + 1;
end

% Evaluating the gradient     
function g = grad(A,c,rho,x,expz)
n = size(A,1);
tmp = 1/n./(ones(n,1)+1./expz);
g = -A'*tmp + rho*(x-c); 


% Finding the stepsize
function [x_new,f_new,g_new] = linesearch(A,c,rho,x,d,delta,f_max,gamma)
alphas = 1;
while (1)
  x_new = x + alphas*d;
  [f_new,expf] = fun(A,x_new);
  f_new = f_new + 0.5*rho*norm(x_new-c)^2;
  if (f_new <= f_max + gamma*alphas*delta) || (alphas <= 1e-8)
    break
  else
    alphas = alphas/2;
  end
end
g_new = grad(A,c,rho,x_new,expf);


  
