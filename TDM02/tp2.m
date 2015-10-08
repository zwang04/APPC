% tp2

clear all
close all
clc
rand('seed',3);
randn('seed',3);

%%

n = 50;
p = 18;
Xi = randn(n,p);
C = corrcoef(Xi);
Xi = Xi*C;
Xi = (Xi - ones(n,1)*mean(Xi))./(ones(n,1)*std(Xi,1));
Xi = Xi/sqrt(n);
betaVrai = zeros(p,1);
betaVrai(1:10) = [1 2 3 4 5 1 2 3 4 5];
sig = 0.25;
yi = Xi*betaVrai + sig*randn(n,1);
[n,p] = size(Xi);
tic
b_mc = (Xi'*Xi)\(Xi'*yi);
ErrMC = norm(betaVrai-b_mc);
       % moindre carree ??? qui est choisi
toc      

%% 2

lambda = 2;
tic
cvx_begin
variables betaL(p)
minimize( .5* (yi - Xi*betaL)'*(yi - Xi*betaL) + lambda * sum(abs(betaL)) );
cvx_end

Err = norm(betaVrai-betaL);
fprintf('Lasso Primal: %10.2f - ', Err)
toc

tic
cvx_begin;
variables beta_dual(p);
dual variable d;
minimize( norm(Xi*beta_dual,2) );
subject to;
d : abs(Xi'*(Xi*beta_dual - yi)) <= lambda;
cvx_end;
Err = norm(betaVrai-beta_dual);
fprintf('Lasso Dual: %10.2f - ', Err)
toc


tic
cvx_begin
variables alpha_cvx(n)
dual variable d1
minimize( .5*alpha_cvx'*alpha_cvx - alpha_cvx'*yi );
subject to
d1 : abs(Xi'*alpha_cvx) <= lambda;
cvx_end
beta_alpha = (Xi'*Xi)\Xi'*(yi+alpha_cvx);
ErrLasso = norm(betaVrai-beta_alpha);
fprintf('Lasso Dual: %10.2f - ', ErrLasso)
toc

% gurobi pour la semaine prochaine

% clear model;
% model.Q = .5*sparse(Xi'*Xi);
% model.obj = zeros(p,1);
% model.A = [sparse(Xi'*Xi);sparse(Xi'*Xi)];
% model.sense = [repmat('<',p,1);repmat('>',p,1)];
% clear params;
% params.Presolve = 2;
% params.TimeLimit = 100;
% params.OutputFlag = 0;
% model.rhs = [lambda + Xi'*yi;-lambda + Xi'*yi];
% result = gurobi(model, params);
% beta_gu = result.x;

%% 3

mu = 1;
mu = .5;% mieux que mu = 1

tic
cvx_begin;
variables beta_en(p);
minimize( .5* (yi - Xi*beta_en)'*(yi - Xi*beta_en) + lambda * sum(abs(beta_en)) + mu/2*norm(beta_en));
cvx_end;
Err = norm(betaVrai-beta_en);
fprintf('El Net Primal: %10.2f - ', Err)
toc

tic
beta_enL = monLasso([Xi ; sqrt(mu)*eye(p)],[yi;zeros(p,1)]);
Err = norm(betaVrai-beta_enL);
fprintf('El Net Primal: %10.2f - ', Err)
toc

w = 1./abs(b_mc); % set the weight

tic
cvx_begin
variables beta_adapt_lasso(p)
minimize( .5* (yi - Xi*beta_adapt_lasso)'*(yi - Xi*beta_adapt_lasso) + lambda * w'*abs(beta_adapt_lasso) )
cvx_end
ErrAdaptiveLasso = norm(betaVrai-beta_adapt_lasso);
fprintf('Adaptive Primal: %8.2f - ', ErrAdaptiveLasso)

% Itératif 
tic
temp = monLasso(Xi*diag(sqrt(w)),yi);

while( norm(temp - w) > 0.01) 
    temp = w ;
    beta_adL_iterative = monLasso(Xi*diag(sqrt(w)),yi);
    w = 1./abs(beta_adL_iterative);
end
toc
ErrAdaptiveLassoIterative = norm(betaVrai-beta_adL_iterative);
fprintf('Iterative : %8.2f -',ErrAdaptiveLassoIterative)
 
% astuce : B = B+ - B-

w = 1./abs(b_mc); % set the weight
%c

b = 2;
tic
cvx_begin
variables beta_cvx(p) beta_cvxm(p)
dual variable d
minimize( norm(yi - Xi*(beta_cvx-beta_cvxm),2) )
subject to
d : w'*(beta_cvx+beta_cvxm) <= b;
beta_cvx >= 0;
beta_cvxm >= 0;
cvx_end
fprintf('CVX Primal : ')
toc
beta_cvx = beta_cvx-beta_cvxm;

% avec monqp

%d
tic
H = [Xi'*Xi -Xi'*Xi;-Xi'*Xi Xi'*Xi];
c = [Xi'*yi ; -Xi'*yi];
l = 10^-9;
A = [w ; w];
b = w'*abs(beta_adapt_lasso);
[xnew, dual_var, pos] = monqp(H,c,A,b,inf,l,0);
fprintf('Adaptive monQP: ')
toc
b_mqp = zeros(2*p,1);
b_mqp(pos) = xnew;
b_mqp = b_mqp(1:p) - b_mqp(p+1:end);

% avec monLasso

%e
tic
beta_adL = monLasso(Xi*diag(sqrt(w)),yi);
Err = norm(betaVrai-beta_adL);
fprintf('Adaptive monLasso: %6.2f - ', Err)
toc

% dual 

% f
tic
cvx_begin
cvx_precision best
variables alpha_cvx(n)
dual variable d
minimize( .5*alpha_cvx'*alpha_cvx - alpha_cvx'*yi )
subject to
d : abs(Xi'*alpha_cvx) <= lambda*w;
cvx_end
fprintf('CVX Dual: ')
toc
beta_alpha = (Xi'*Xi)\Xi'*(yi-alpha_cvx);

% g
tic
cvx_begin
cvx_precision best
variables beta_dual(p)
dual variable d
minimize( norm(Xi*beta_dual,2) )
subject to
d : abs(Xi'*(Xi*beta_dual - yi)) <= lambda*w;
cvx_end
fprintf('CVX Dual 2: ')
toc
