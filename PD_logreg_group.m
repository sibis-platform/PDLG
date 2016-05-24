% This program aims to solve the problem:
% min e'*f(Ax)/n
% s.t. Group cardinality of x is less than or equal to k,
% where e is the all-ones vector, f(y) = log(1+exp(-y)),
% A = [b1*z1 b1; b2*z2 b2; ... bn*zn bn],
% and label bi in {-1,1} and sample data zi in Re^p for i=1, ..., n. 
%        
% by applying penalty decomposition method to the problem:
% min e'*f(Ax)/n
% s.t. y - A*x = 0, 
%      Group cardinality of y is less than or equal to k,
%
%------ Required input ------
%
% Z     - the n x p data matrix
% b     - binary outcomes
% k     - the desired cardinality (i.e., the number of nonzero groups)
% group - the grouping information; each cell contains the indices for each group
%
%------ Optional input ------
% tol   - the tolerance for termination (Default: 1e-4)
% maxit - the maximum of number of iterations for running code (Default: 1000)
% init  - initial point (Default: a randomly generated feasible point)
%
%------ output ------
%
% x     - approximate sparse solution; last component is the bias term
%

function x = PD_logreg_group(Z,b,k,group,varargin)

% Preprocess data
[n,p] = size(Z);
N_group = length(group);

mu = mean(Z);
sigma = zeros(p,1);
for (i = 1:p)
    tmp = Z(:,i) - mu(i);
    sigma(i) = sqrt(tmp'*tmp/(n-1));
end

for i = 1:p
    if sigma(i) ~= 0
        Z(:,i) = (Z(:,i) -mu(i))/sigma(i);
    end
end
A = [diag(b)*Z b];

% Initialization
p = p +1;

I = randperm(N_group);
x0 = zeros(p-1,1);
for i = 1:k
    x0(group{I(i)}) = rand(length(group{I(i)}),1);
end
x0 = [x0;1];

pars.tol = 1e-4;
pars.maxit = 1000;
pars.init = x0;


if(length(varargin)>=1)
    if(~isstr(varargin{1}))
        pars=varargin{1};
        for j=1:length(varargin)-1
            varargin{j}=varargin{j+1};
        end;
    end;
    for i=1:nargin-4
        if(isstr(varargin{i}))
            if(i+1>nargin-4 || isstr(varargin{i+1})) 
                val=1;
            else
                val=varargin{i+1};
            end;
            eval(['pars.' varargin{i} '=val;']);
        end;
    end;
end;


eps = pars.tol;
maxit = pars.maxit;
z = pars.init;
x = pars.init; x_old = x; y = x;
rho = 0.1; iter = 1; tol = 1e-3; 

% Main algorithm
while 1==1 
  best_obj = inf;
  while (iter <= maxit)
    % solve y 
    [y,z] = SPGsolver(A,x,rho,y,z);
    % solve x
    groupy = zeros(N_group,1);
    for i = 1:N_group
        ind = group{i};
        groupy(i) = norm(y(ind));
    end
    [tmp,I] = sort(groupy,'descend');
    x = zeros(p,1);
    for i = 1:k
        x(group{I(i)}) = y(group{I(i)});
    end
    x(p) = y(p);
    
    % Updating error, residue, and objective information
    err = norm(x-x_old,inf)/(norm(x,'inf')+1);
    obj = fun(A,y);
    obj = obj + rho*norm(x-y)^2/2;
    res = norm(x-y,inf);
    iter = iter + 1; 
    
    %Check the stopping criterion for the inner loop
    if err <= tol
        if best_obj - obj > (1e-3)*abs(best_obj) || (best_obj == inf && obj < best_obj)
            % heuristics for improving the quality of the solution 
            best_obj = obj;
            best_k = 0;
            for i = 1:N_group
                ind = group{i};
                if norm(x(ind)) ~= 0
                    best_k = best_k + 1;
                end
            end
            best_x = x; best_res = res; r = 1;
        end        
        if r >= min(2,best_k+1) || rho > 1;
            break;
        end;
        groupy = zeros(N_group,1);
        for i = 1:N_group
            ind = group{i};
            groupy(i) = norm(best_x(ind));
        end
        [tmp,I] = sort(groupy,'descend');
        x = best_x;
        x(group{I(best_k-r+1)}) = 0;
        r = r+1;
    end
    x_old = x;
    
    % Display algorithm information
    if mod(iter,500)==0
        fprintf('iter=%03d res = %2.8f err = %2.8f obj = %2.8f\n',iter,res,err,obj);   
    end
  end
  
  % Checking the stopping criterion for the outer loop
  if res < 1e-3 || iter > maxit; break; end;  
  
  % Updating the penalty and outer loop tolerance parameter
  if best_obj < obj 
    x = best_x; res = best_res;
  end
  rho = min(sqrt(10)*rho,1e15);
  tol = max(tol/sqrt(10), eps);
  y = x; x_old = x;
end

% Postprocess
x(1:p-1) = x(1:p-1)./max(1e-8,sigma);
x(p) = x(p) - mu*x(1:p-1);



