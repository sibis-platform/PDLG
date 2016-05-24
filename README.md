## Penalty Decomposition (PD) method for solving group cardinality constrained logistic regression.

This package contains the MATLAB source codes for the penalty decomposition method solving 
the group cardinality constrained logistic regression problem of the following paper:

Yong Zhang, Dongjin Kwon and Kilian M. Pohl:
"Computing Group Cardinality Constraint Solutions for Logistic Regression Problems," MEDIA, 2016

When using the code, please cite the paper and the DOI: https://dx.doi.org/10.6084/m9.figshare.3398332

The software development was supported by NIH grants (R01 HL127661, K05 AA017168) and the Creative and Novel Ideas in HIV Research (CNIHR) Program through a supplement to the University of Alabama at Birmingham (UAB) Center For AIDS Research funding (P30 AI027767). This funding was made possible by collaborative efforts of the Office of AIDS Research, the National Institute of Allergy and Infectious Diseases, and the International AIDS Society. 


To run the codes, the users should properly input data and parameters for the function PD_logreg_group as described below:

```
% -----  Function:  ----- 
%
% function x = PD_logreg_group(Z,b,k,group,varargin)
% 
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
```

An example file for how to set up the input is given in test.m which is including
in the package. For any questions, please contact Dr. Zhang via michaelzhang917@gmail.com


Yong Zhang, Dongjin Kwon and Kilian Pohl

See LICENSE file distributed along with the package for the copyright and license terms.

Last updated on May 24th, 2016
