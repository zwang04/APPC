%% tp6
close all
clear all
clc

n= 10;
p=50;

d=p;

beta= randn(p,1);
beta(1:10)=0;
X=randn(n,p);
y=X*beta;

lambda=.1;
w=zeros(d,1);
stepsize = 1/norm(X'*X); % choose L as a step, L being the norm of the Hessian
for i=1:5000
    grad= -X'*(y-X*w);
    w=w - stepsize*grad;
    w=proxl1(w,stepsize*lambda);
end;

epsi=1e-6;
indzero=find(abs(w)<epsi);
indnonzero=find(abs(w)>=epsi);
grad=-X'*(y-X*w);
exactOnZeros= max(abs(grad(indzero))-lambda);
exactOnNonZeros= max( abs(abs(grad(indnonzero)) - lambda));

lambda=.1;
epsi=1e-3;
tic

wprox=ProximalSparseRegression(X,y,lambda,epsi);

timingprox=toc
tic
xcd=CDsparseRegression(X,y,lambda,epsi);
timingcoordinate=toc


% loading the data X y (training) xtest, ytest (test)
load('housing.mat');
% choose the penalty value. the larger lambda, the sparser the solution
lambda=1;
% compute stepsize $L$ the norm of the Hessian
augmentedMatrix=[X ones(size(X,1),1)];
stepsize=1/norm(augmentedMatrix'*augmentedMatrix);
w=zeros(size(X,2),1);
w0=0;
% proximal descent algorithm
Y=diag(y);
for i=1:5000
% computing the gradient wrt w and w0
loss=max(1-Y*(X*w + w0),0);
gradw=-(Y*X)'*loss;
gradw0=-(Y*ones(size(X,1),1))'*loss;
% proximal step
w=proxl1(w-stepsize*gradw,lambda*stepsize);
w0=w0-stepsize*gradw0;
end;
% check how good your algorithm is doing on the test set
mean(sign(xtest*w+w0)==ytest)