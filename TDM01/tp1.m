%% tp1

%% 1.a
prepare_housing % load data
randn('seed',10);
q = 5;
% adding useless variables
X = [X randn(n,q)];

%% 1.b
ind = randperm(n);
na = n/2;
Xi = X(ind(1:na),:);
yi = y(ind(1:na));
Xt = X(ind(na+1:end),:);
yt = y(ind(na+1:end));


%% 1.c
beta_mc = Xi\yi; % matrix divided by vector

Erreur_app = (yi - Xi*beta_mc)'*(yi - Xi*beta_mc)
Erreur = (yt - Xt*beta_mc)'*(yt - Xt*beta_mc)

%% 2.a

p = 18; % size of beta

k = 3; % 1 2 3 .... 10
cvx_begin
cvx_precision best
variables beta1(p)
dual variable d
minimize( norm(yi - Xi*beta1,2) )
subject to
d : sum(abs(beta1)) <= k;
cvx_end

%% 2.b

cvx_begin
variables beta2(p)
minimize( norm(yi - Xi*beta2,2) + d * sum(abs(beta2)) )
cvx_end

%% 2.c
% 
% D = Xi'*Xi;
% ep = -yi'*Xi;
% cvx_begin
% variables beta2(p)
% dual variable d2
% minimize( .5*beta2'*D*beta2 + d2*beta2 )
% subject to
% d2 : sum(abs(beta2)) <= k;
% cvx_end

%% 2.d

H = [Xi'*Xi -Xi'*Xi;-Xi'*Xi Xi'*Xi];
c = [Xi'*yi ; -Xi'*yi];
A = ones(2*p,1);
b = k;
l = 10^-12;
verbose = 0;

%% 2.e

cvx_begin
variables Bpm(2*p)
dual variable dpm
minimize( .5*Bpm'*H*Bpm - c'*Bpm )
subject to
dpm : sum(Bpm) <= b;
0 <= Bpm;
cvx_end

%% 2.f

[xnew, lambda, pos] = monqp(H,c,A,k,inf,l,verbose);

%% 2.g

ind = find(pos>p);
sign = ones(length(pos),1);
pos(ind) = pos(ind) - p;
sign(ind) = -1;
betam = 0*beta1;
betam(pos) = sign.*xnew;
[beta1 beta2 Bpm(1:p)-Bpm(p+1:end) betam]


%% 3
beta_mc = Xi\yi;
err_mc = (yt - Xt*beta_mc)'*(yt - Xt*beta_mc);
err_L = (yt - Xt*beta1)'*(yt - Xt*beta1);

pos = find(abs(beta1)>0.000001);
beta_mc = Xi(:,pos)\yi;
err_Lmc = (yt - Xt(:,pos)*beta_mc)'*(yt - Xt(:,pos)*beta_mc);

[err_mc err_L err_Lmc]

%% 4 


K = [2:0.25:30];
B = [];
for i=1:length(K)
k = K(i);
[xnew, lambda, pos] = monqp(H,c,A,k,inf,l,verbose);
ind = find(pos>p);
sign = ones(length(pos),1);
pos(ind) = pos(ind) - p;
sign(ind) = -1;
%I =eye(length(pos));
beta_mc = Xi(:,pos)\yi;
beta_mc = Xi(:,pos)\yi;
err(i) = (yt - Xt(:,pos)*beta_mc)'*(yt - Xt(:,pos)*beta_mc);
beta = 0*beta1;
beta(pos) = beta_mc;
B = [B beta];
end;

[v ind] = min(err);
beta = B(:,ind);
ind = find(abs(beta) < 0.000001);
Xi(:,ind) = [];
Xt(:,ind) = [];
beta_mc = Xi\yi;
Erreur = (yt - Xt*beta_mc)'*(yt - Xt*beta_mc)


% close all
% figure(1)
% subplot(2,1,1)
% plot(K,err)
% subplot(2,1,2)
% plot(K,B')

%%

figure(2)
monLasso(X,y)

