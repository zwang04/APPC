%% tp5

close all
clear all
clc

% setup cxv

%% 
lambda = 1.5;
gam = 2;
res = [];
n= 10;
p=50;

% test count_mcp
b = (-4:0.1:4);
for i=1:length(b)
    Beta = b(i);
    cout = cout_mcp(zeros(n,1), zeros(n,1),Beta,lambda,gam);
    res(i) = cout;
end
figure(1)
plot(b,res)

%test grad_mcp
n= 10;
p=50;

beta= randn(p,1);
beta(1:10)=0;
X=randn(n,p);
y=X*beta;

%Beta = (0:lambda*gam -1 : gam*lambda-1 : 1/2*lambda*gam);
chouia =10^-6;
d = eye(p);
c0 = cout_mcp(X, y,beta,lambda,gam);
g= grad_mcp(X,y,beta,lambda,gam);
for i =1:p
    ce = cout_mcp(X,y,beta+chouia*d(:,1),lambda,gam);
    err(i) = g(i) - (ce -c0)/chouia;
end
figure(2)
plot(err)

% test grd_mcp