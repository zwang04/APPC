function [ beta ] = monLasso( X,y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n,p] = size(X);
%X = (X - ones(n,1)*mean(X))./(ones(n,1)*std(X));

%%

ind = randperm(n);
na = round(n/2);
Xi = X(ind(1:na),:);
yi = y(ind(1:na));

Xt = X(ind(na+1:end),:);
yt = y(ind(na+1:end));

%%

H = [Xi'*Xi -Xi'*Xi;-Xi'*Xi Xi'*Xi];
c = [Xi'*yi ; -Xi'*yi];
A = ones(2*p,1);
l = 10^-8;
verbose = 0;
 
K = [2:0.5:30];  % Attention - ce doit �tre adapt� en fonction de X et y

 B  = [];
 for i=1:length(K)
     
     k = K(i);
     [xnew, lambda, pos] = monqp(H,c,A,k,inf,l,verbose);
     
     ind = find(pos>p);
     sign = ones(length(pos),1); 
     pos(ind) = pos(ind) - p;
     sign(ind) = -1;
     I = eye(length(pos));
     beta_mc = (Xi(:,pos)'*Xi(:,pos)+ l*I)\(Xi(:,pos)'*yi);
     
     err(i) = (yt - Xt(:,pos)*beta_mc)'*(yt - Xt(:,pos)*beta_mc);
     beta = zeros(p,1);
     beta(pos) = beta_mc;
     B = [B beta];

 end;
 
 %%
 
 [v ind] = min(err);
 k = K(ind);

H = [X'*X -X'*X;-X'*X X'*X];
c = [X'*y ; -X'*y];
A = ones(2*p,1);
l = 10^-8;
[xnew, lambda, pos] = monqp(H,c,A,k,inf,l,verbose);
ind1 = find(pos<=p);
ind = find(pos>p);
pos = [pos(ind1) ; pos(ind)-p] ;
X = X(:,pos);
 betaL = X\y;
 beta = zeros(p,1);  
 beta(pos) = betaL;


end

